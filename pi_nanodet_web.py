#!/usr/bin/env python3
"""
Deteccion de drones con NanoDet-Plus (NCNN) en Raspberry Pi 4B + Camera Module V3.
Muestra el video con las cajas y los FPS en el navegador.

Uso:
  python3 pi_nanodet_web.py                        # camara + video en http://<IP-de-la-Pi>:8000
  python3 pi_nanodet_web.py --image prueba.jpg     # validar con una imagen, sin camara
  python3 pi_nanodet_web.py --image prueba.jpg --bench 100   # medir inferencia pura

Decodificacion verificada contra nanodet/model/head/nanodet_plus_head.py:
  - el sigmoid de las clases va DENTRO del grafo exportado (_forward_onnx)
  - salida [N, num_classes + 4*(reg_max+1)], niveles en orden de stride [8, 16, 32, 64]
  - centro de cada anchor = (columna*stride, fila*stride), SIN desplazamiento de medio pixel
  - distancia = integral de la distribucion softmax * stride; caja = centro -/+ distancia
"""
import argparse
import json
import math
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import ncnn

# Normalizacion del config de NanoDet-Plus, en orden BGR (igual que en el entrenamiento)
MEAN = [103.53, 116.28, 123.675]
NORM = [1 / 57.375, 1 / 57.12, 1 / 58.395]
STRIDES = [8, 16, 32, 64]
REG_MAX = 7


# ============================================================================ modelo
class NanoDet:
    def __init__(self, param, binf, size=320, threads=4, thr=0.35, iou=0.6,
                 in_blob="in0", out_blob="out0"):
        self.size, self.thr, self.iou = size, thr, iou
        self.in_blob, self.out_blob = in_blob, out_blob

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = threads
        self.net.opt.use_packing_layout = True
        if self.net.load_param(param) != 0:
            raise RuntimeError(f"no se pudo cargar {param}")
        if self.net.load_model(binf) != 0:
            raise RuntimeError(f"no se pudo cargar {binf}")

        # centros de los anchors, en el mismo orden que el exportador (fila mayor)
        cx, cy, st = [], [], []
        for s in STRIDES:
            n = math.ceil(size / s)
            fila, col = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            cx.append(col.ravel() * s)
            cy.append(fila.ravel() * s)
            st.append(np.full(n * n, s))
        self.cx = np.concatenate(cx).astype(np.float32)
        self.cy = np.concatenate(cy).astype(np.float32)
        self.stride = np.concatenate(st).astype(np.float32)
        self.proyeccion = np.arange(REG_MAX + 1, dtype=np.float32)
        self._forma_verificada = False

    def _inferir(self, img):
        """img: BGR uint8 de size x size. Devuelve la salida cruda [N, canales]."""
        m = ncnn.Mat.from_pixels(img, ncnn.Mat.PixelType.PIXEL_BGR, self.size, self.size)
        m.substract_mean_normalize(MEAN, NORM)
        ex = self.net.create_extractor()
        ex.input(self.in_blob, m)
        _, out = ex.extract(self.out_blob)
        return np.array(out)

    def detectar(self, bgr):
        """bgr: frame de cualquier tamano. Devuelve (detecciones, tiempos_ms).
        Cada deteccion: [x1, y1, x2, y2, score, clase] en coordenadas del frame."""
        h, w = bgr.shape[:2]
        t0 = time.perf_counter()
        # keep_ratio=False en el entrenamiento: se estira al cuadrado, sin bandas
        img = cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        t1 = time.perf_counter()
        out = self._inferir(img)
        t2 = time.perf_counter()
        dets = self._decodificar(out, w / self.size, h / self.size)
        t3 = time.perf_counter()
        return dets, {"pre": (t1 - t0) * 1e3, "inf": (t2 - t1) * 1e3, "post": (t3 - t2) * 1e3}

    def _decodificar(self, out, sx, sy):
        if not self._forma_verificada:
            esperado = len(self.cx)
            if out.ndim != 2 or out.shape[0] != esperado or out.shape[1] <= 4 * (REG_MAX + 1):
                raise RuntimeError(
                    f"salida con forma {out.shape}, se esperaba ({esperado}, clases+32). "
                    f"Revisa --size (debe ser el del entrenamiento) y los nombres de blob.")
            self._forma_verificada = True

        nc = out.shape[1] - 4 * (REG_MAX + 1)
        scores = out[:, :nc]
        clase = scores.argmax(1)
        conf = scores[np.arange(len(scores)), clase]
        idx = np.flatnonzero(conf > self.thr)
        if idx.size == 0:
            return np.zeros((0, 6), np.float32)
        if idx.size > 200:  # acotar el costo del NMS
            idx = idx[np.argsort(conf[idx])[::-1][:200]]

        reg = out[idx, nc:].reshape(-1, 4, REG_MAX + 1)
        reg = np.exp(reg - reg.max(-1, keepdims=True))
        reg /= reg.sum(-1, keepdims=True)
        dist = (reg @ self.proyeccion) * self.stride[idx, None]      # [n, 4] = l, t, r, b

        cx, cy = self.cx[idx], self.cy[idx]
        cajas = np.stack([cx - dist[:, 0], cy - dist[:, 1],
                          cx + dist[:, 2], cy + dist[:, 3]], 1).clip(0, self.size)
        cf, cl = conf[idx], clase[idx]

        # NMS por clase: desplazar cada clase a su propia region del plano
        keep = nms(cajas + (cl * (self.size + 1))[:, None], cf, self.iou)
        cajas = cajas[keep] * np.array([sx, sy, sx, sy], np.float32)
        return np.column_stack([cajas, cf[keep], cl[keep]]).astype(np.float32)


def nms(cajas, scores, iou):
    x1, y1, x2, y2 = cajas.T
    areas = (x2 - x1) * (y2 - y1)
    orden = scores.argsort()[::-1]
    keep = []
    while orden.size:
        i = orden[0]
        keep.append(i)
        resto = orden[1:]
        iw = np.clip(np.minimum(x2[i], x2[resto]) - np.maximum(x1[i], x1[resto]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[resto]) - np.maximum(y1[i], y1[resto]), 0, None)
        inter = iw * ih
        orden = resto[inter / (areas[i] + areas[resto] - inter + 1e-9) <= iou]
    return np.array(keep, dtype=np.int64)


# ============================================================================ dibujo
def dibujar(frame, dets, nombres, stats=None):
    img = frame.copy()
    h, w = img.shape[:2]
    for x1, y1, x2, y2, sc, cl in dets:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0, 140, 255), 2)
        nombre = nombres[int(cl)] if int(cl) < len(nombres) else str(int(cl))
        etiqueta = f"{nombre} {sc * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), (0, 140, 255), -1)
        cv2.putText(img, etiqueta, (p1[0] + 2, p1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # mira en el centro y linea al objetivo principal
    c = (w // 2, h // 2)
    cv2.drawMarker(img, c, (200, 200, 200), cv2.MARKER_CROSS, 20, 1)
    if len(dets):
        b = dets[dets[:, 4].argmax()]
        cv2.line(img, c, (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)), (0, 255, 0), 1)

    if stats:
        texto = (f"FPS {stats['fps']:.1f} | inferencia {stats['inf_ms']:.0f} ms"
                 f" | camara {stats['cam_fps']:.0f} fps")
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (0, 0), (tw + 12, th + 12), (0, 0, 0), -1)
        cv2.putText(img, texto, (6, th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 1, cv2.LINE_AA)
    return img


# ============================================================================ camara
class Camara:
    """Captura en un hilo propio: la espera del siguiente frame se solapa con la inferencia."""

    def __init__(self, w, h, fps, hflip=False, vflip=False):
        from picamera2 import Picamera2
        from libcamera import Transform, controls

        self.picam = Picamera2()
        sensor = self._modo_campo_completo()
        cfg = self.picam.create_video_configuration(
            main={"size": (w, h), "format": "RGB888"},   # RGB888 de Picamera2 = bytes B,G,R
            transform=Transform(hflip=int(hflip), vflip=int(vflip)),
            buffer_count=4,
            controls={"FrameRate": fps},
            **({"sensor": sensor} if sensor else {}))
        self.picam.configure(cfg)
        self.picam.start()

        # foco fijo en infinito y exposicion corta (menos desenfoque por movimiento)
        for ctrl in ({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.0},
                     {"AeExposureMode": controls.AeExposureModeEnum.Short}):
            try:
                self.picam.set_controls(ctrl)
            except Exception:
                pass  # camaras sin autofoco o sin ese modo

        self.frame, self.seq, self.cam_fps = None, 0, 0.0
        self.cond = threading.Condition()
        self.activo = True
        threading.Thread(target=self._bucle, daemon=True).start()

    def _modo_campo_completo(self):
        """Elige el modo de sensor mas rapido que cubre todo el sensor (sin recorte).
        En la IMX708 es 2304x1296 binning 2x2; el modo 1536x864 recorta el centro."""
        try:
            modos = self.picam.sensor_modes
            ancho = max(m["crop_limits"][2] for m in modos)
            completos = [m for m in modos if m["crop_limits"][2] == ancho]
            m = max(completos, key=lambda m: m["fps"])
            print(f"Modo de sensor: {m['size']} a {m['fps']:.0f} fps, campo completo")
            return {"output_size": m["size"], "bit_depth": m["bit_depth"]}
        except Exception as e:
            print("No se pudo elegir modo de sensor, se usa el automatico:", e)
            return None

    def _bucle(self):
        n, t0 = 0, time.perf_counter()
        while self.activo:
            f = self.picam.capture_array("main")
            with self.cond:
                self.frame, self.seq = f, self.seq + 1
                self.cond.notify_all()
            n += 1
            t = time.perf_counter()
            if t - t0 >= 1.0:
                self.cam_fps, n, t0 = n / (t - t0), 0, t

    def leer(self, ultimo):
        """Espera un frame mas nuevo que 'ultimo' y lo devuelve junto con su numero."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq > ultimo, timeout=2.0)
            if self.frame is None:
                raise RuntimeError("la camara no entrega frames")
            return self.frame, self.seq

    def cerrar(self):
        self.activo = False
        time.sleep(0.1)
        self.picam.stop()


# ============================================================================ servidor web
class Estado:
    def __init__(self):
        self.cond = threading.Condition()
        self.frame = self.dets = None
        self.stats = {}
        self.res_seq = 0
        self.jpeg, self.jpeg_seq = None, 0
        self.clientes = 0

    def publicar(self, frame, dets, stats):
        with self.cond:
            self.frame, self.dets, self.stats = frame, dets, stats
            self.res_seq += 1
            self.cond.notify_all()


def hilo_codificador(estado, nombres, max_fps, calidad):
    """Dibuja y comprime a JPEG solo si hay alguien mirando, en un hilo aparte
    para no frenar la inferencia (cv2 libera el GIL mientras trabaja)."""
    visto, ultimo = 0, 0.0
    params = [cv2.IMWRITE_JPEG_QUALITY, calidad]
    while True:
        with estado.cond:
            estado.cond.wait_for(lambda: estado.res_seq > visto and estado.clientes > 0)
            visto = estado.res_seq
            frame, dets, stats = estado.frame, estado.dets, dict(estado.stats)
        espera = 1.0 / max_fps - (time.perf_counter() - ultimo)
        if espera > 0:
            time.sleep(espera)
        ultimo = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", dibujar(frame, dets, nombres, stats), params)
        if ok:
            with estado.cond:
                estado.jpeg, estado.jpeg_seq = buf.tobytes(), estado.jpeg_seq + 1
                estado.cond.notify_all()


PAGINA = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deteccion de drones</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:15px system-ui,sans-serif;text-align:center}
 img{width:100%;max-width:960px;display:block;margin:12px auto;background:#000}
 #s{font-family:monospace;padding:6px}
</style></head><body>
<h3>NanoDet-Plus &middot; Raspberry Pi 4B</h3>
<img src="/stream" alt="video">
<div id="s">conectando...</div>
<script>
async function tick(){
  try{
    const s = await (await fetch('/stats')).json();
    document.getElementById('s').textContent =
      `FPS ${s.fps.toFixed(1)} | pre ${s.pre_ms.toFixed(1)} ms | inferencia ${s.inf_ms.toFixed(1)} ms` +
      ` | post ${s.post_ms.toFixed(1)} ms | camara ${s.cam_fps.toFixed(0)} fps` +
      ` | detecciones ${s.n} | offset (${s.dx.toFixed(2)}, ${s.dy.toFixed(2)}) | CPU ${s.temp} C`;
  }catch(e){}
  setTimeout(tick, 500);
}
tick();
</script></body></html>"""


def crear_manejador(estado):
    class Manejador(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                cuerpo = PAGINA.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(cuerpo)))
                self.end_headers()
                self.wfile.write(cuerpo)
            elif self.path == "/stats":
                with estado.cond:
                    cuerpo = json.dumps(estado.stats).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(cuerpo)
            elif self.path == "/stream":
                self._stream()
            else:
                self.send_error(404)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with estado.cond:
                estado.clientes += 1
                estado.cond.notify_all()
            visto = 0
            try:
                while True:
                    with estado.cond:
                        estado.cond.wait_for(lambda: estado.jpeg_seq > visto, timeout=2.0)
                        if estado.jpeg_seq == visto:
                            continue
                        visto, jpg = estado.jpeg_seq, estado.jpeg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpg)).encode() +
                                     b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with estado.cond:
                    estado.clientes -= 1

    return Manejador


# ============================================================================ utilidades
def ip_local():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))   # no envia nada, solo elige la interfaz de salida
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "IP-de-la-Pi"


def temperatura():
    try:
        return round(int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000, 1)
    except OSError:
        return 0.0


def cargar_nombres(ruta, param):
    for p in (ruta, os.path.join(os.path.dirname(os.path.abspath(param)), "classes.json")):
        if p and os.path.exists(p):
            return json.load(open(p))
    return ["drone"]


# ============================================================================ modos
def modo_imagen(det, nombres, ruta, bench):
    img = cv2.imread(ruta)   # BGR, igual que en el entrenamiento
    if img is None:
        raise SystemExit(f"no se pudo leer {ruta}")
    dets, t = det.detectar(img)
    print(f"{ruta}: {img.shape[1]}x{img.shape[0]} | {len(dets)} detecciones | "
          f"pre {t['pre']:.1f} ms, inferencia {t['inf']:.1f} ms, post {t['post']:.1f} ms")
    for x1, y1, x2, y2, sc, cl in dets:
        print(f"  {nombres[int(cl)]:8s} {sc:.3f}  caja=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
    salida = os.path.splitext(ruta)[0] + "_det.jpg"
    cv2.imwrite(salida, dibujar(img, dets, nombres))
    print("Imagen con cajas guardada en", salida)

    if bench:
        det.detectar(img)  # calentamiento
        ts = [det.detectar(img)[1]["inf"] for _ in range(bench)]
        print(f"Inferencia pura ({bench} corridas): media {np.mean(ts):.1f} ms, "
              f"mediana {np.median(ts):.1f} ms -> {1000 / np.mean(ts):.1f} FPS maximos")


def modo_camara(det, a, nombres):
    estado = Estado()
    cam = Camara(a.width, a.height, a.cam_fps, a.hflip, a.vflip)
    threading.Thread(target=hilo_codificador,
                     args=(estado, nombres, a.stream_fps, a.jpeg), daemon=True).start()
    servidor = ThreadingHTTPServer(("0.0.0.0", a.port), crear_manejador(estado))
    servidor.daemon_threads = True
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    print(f"\nAbre en el navegador:  http://{ip_local()}:{a.port}\n(Ctrl+C para salir)\n")

    seq, fps, prev, ultimo_log = 0, 0.0, time.perf_counter(), 0.0
    try:
        while True:
            frame, seq = cam.leer(seq)
            dets, t = det.detectar(frame)

            ahora = time.perf_counter()
            inst = 1.0 / max(ahora - prev, 1e-6)
            prev = ahora
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst   # media movil

            dx = dy = 0.0
            if len(dets):
                b = dets[dets[:, 4].argmax()]
                h, w = frame.shape[:2]
                dx = ((b[0] + b[2]) / 2 - w / 2) / (w / 2)   # -1 izquierda .. +1 derecha
                dy = ((b[1] + b[3]) / 2 - h / 2) / (h / 2)   # -1 arriba .. +1 abajo

            stats = {"fps": fps, "pre_ms": t["pre"], "inf_ms": t["inf"], "post_ms": t["post"],
                     "cam_fps": cam.cam_fps, "n": int(len(dets)),
                     "dx": float(dx), "dy": float(dy), "temp": temperatura()}
            estado.publicar(frame, dets, stats)

            if ahora - ultimo_log > 2.0:
                ultimo_log = ahora
                print(f"FPS {fps:5.1f} | inferencia {t['inf']:5.1f} ms | "
                      f"camara {cam.cam_fps:4.0f} fps | det {len(dets)} | "
                      f"offset ({dx:+.2f}, {dy:+.2f}) | CPU {stats['temp']} C")
    except KeyboardInterrupt:
        print("\nSaliendo...")
    finally:
        servidor.shutdown()
        cam.cerrar()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--param", default="nanodet_drone_320.ncnn.param")
    ap.add_argument("--bin", default="nanodet_drone_320.ncnn.bin")
    ap.add_argument("--classes", default=None, help="classes.json (por defecto, junto al .param)")
    ap.add_argument("--size", type=int, default=320, help="el MISMO tamano del entrenamiento")
    ap.add_argument("--thr", type=float, default=0.35, help="umbral de confianza")
    ap.add_argument("--iou", type=float, default=0.6, help="umbral IoU del NMS")
    ap.add_argument("--threads", type=int, default=4, help="hilos de NCNN (prueba 3 si hay stream)")
    ap.add_argument("--in-blob", default="in0")
    ap.add_argument("--out-blob", default="out0")
    ap.add_argument("--image", help="probar con una imagen en vez de la camara")
    ap.add_argument("--bench", type=int, default=0, help="con --image: repeticiones para medir")
    ap.add_argument("--width", type=int, default=640, help="ancho del video capturado")
    ap.add_argument("--height", type=int, default=360, help="alto del video capturado")
    ap.add_argument("--cam-fps", type=float, default=50)
    ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--vflip", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--stream-fps", type=float, default=15, help="FPS maximos del video web")
    ap.add_argument("--jpeg", type=int, default=70, help="calidad JPEG del video web")
    a = ap.parse_args()

    det = NanoDet(a.param, a.bin, a.size, a.threads, a.thr, a.iou, a.in_blob, a.out_blob)
    nombres = cargar_nombres(a.classes, a.param)
    print(f"Modelo cargado: {a.param} | entrada {a.size}x{a.size} | clases {nombres} | "
          f"{a.threads} hilos")

    if a.image:
        modo_imagen(det, nombres, a.image, a.bench)
    else:
        modo_camara(det, a, nombres)


if __name__ == "__main__":
    main()
