from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# MediaPipe Pose landmark indices.
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


@dataclass(frozen=True)
class FrameFeatures:
    inverted: float
    stack_error: float
    elbow_bend_deg: float
    knee_bend_deg: float
    shoulder_open_deg: float
    hip_open_deg: float
    rot_proxy: float
    feet_to_hands_norm: float
    arch_score: float
    hollow_score: float


def _lm_xy(landmarks: list[Any], idx: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x, lm.y], dtype=np.float32)


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle_abc(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba) + 1e-6
    nbc = np.linalg.norm(bc) + 1e-6
    cosv = float(np.dot(ba, bc) / (nba * nbc))
    cosv = float(np.clip(cosv, -1.0, 1.0))
    return float(np.arccos(cosv))


def compute_frame_features(landmarks: list[Any], previous_shoulder_angle: float | None) -> tuple[FrameFeatures, float]:
    ls, rs = _lm_xy(landmarks, L_SHOULDER), _lm_xy(landmarks, R_SHOULDER)
    le, re = _lm_xy(landmarks, L_ELBOW), _lm_xy(landmarks, R_ELBOW)
    lw, rw = _lm_xy(landmarks, L_WRIST), _lm_xy(landmarks, R_WRIST)
    lh, rh = _lm_xy(landmarks, L_HIP), _lm_xy(landmarks, R_HIP)
    lk, rk = _lm_xy(landmarks, L_KNEE), _lm_xy(landmarks, R_KNEE)
    la, ra = _lm_xy(landmarks, L_ANKLE), _lm_xy(landmarks, R_ANKLE)

    sh = _mid(ls, rs)
    el = _mid(le, re)
    wr = _mid(lw, rw)
    hp = _mid(lh, rh)
    kn = _mid(lk, rk)
    an = _mid(la, ra)

    shoulder_width = max(_dist(ls, rs), 1e-6)

    shoulder_open = _angle_abc(hp, sh, wr)
    hip_open = _angle_abc(sh, hp, an)
    elbow_angle = _angle_abc(sh, el, wr)
    knee_angle = _angle_abc(hp, kn, an)

    elbow_bend_deg = float(np.degrees(max(0.0, np.pi - elbow_angle)))
    knee_bend_deg = float(np.degrees(max(0.0, np.pi - knee_angle)))
    shoulder_open_deg = float(np.degrees(shoulder_open))
    hip_open_deg = float(np.degrees(hip_open))

    x0 = float(wr[0])
    stack_error = float((abs(float(sh[0]) - x0) + abs(float(hp[0]) - x0) + abs(float(an[0]) - x0)) / shoulder_width)
    inverted = 1.0 if (an[1] < hp[1] < sh[1] < wr[1]) else 0.0

    shoulder_vec = rs - ls
    shoulder_angle = float(np.arctan2(float(shoulder_vec[1]), float(shoulder_vec[0])))
    rot_proxy = 0.0
    if previous_shoulder_angle is not None:
        delta = shoulder_angle - previous_shoulder_angle
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        rot_proxy = abs(float(delta))

    feet_to_hands = (_dist(la, lw) + _dist(ra, rw)) / 2.0
    feet_to_hands_norm = float(feet_to_hands / shoulder_width)

    arch_score = float((float(hp[0]) - float(sh[0])) + (float(hp[0]) - float(an[0])))
    hollow_score = float((float(sh[0]) - float(hp[0])) + (float(an[0]) - float(hp[0])))

    return (
        FrameFeatures(
            inverted=inverted,
            stack_error=stack_error,
            elbow_bend_deg=elbow_bend_deg,
            knee_bend_deg=knee_bend_deg,
            shoulder_open_deg=shoulder_open_deg,
            hip_open_deg=hip_open_deg,
            rot_proxy=rot_proxy,
            feet_to_hands_norm=feet_to_hands_norm,
            arch_score=arch_score,
            hollow_score=hollow_score,
        ),
        shoulder_angle,
    )


def detect_movement(features: list[FrameFeatures]) -> str:
    if not features:
        return "unknown"

    inv = np.array([f.inverted for f in features], dtype=np.float32)
    stack = np.array([f.stack_error for f in features], dtype=np.float32)
    rot = np.array([f.rot_proxy for f in features], dtype=np.float32)
    feet_hands = np.array([f.feet_to_hands_norm for f in features], dtype=np.float32)
    arch = np.array([f.arch_score for f in features], dtype=np.float32)
    hollow = np.array([f.hollow_score for f in features], dtype=np.float32)

    handstand_score = float(np.mean(inv) * (1.0 - min(1.0, float(np.mean(stack)))))
    swing_score = float(np.clip(np.mean(rot) * 8.0 + np.mean(rot > 0.05) * 0.4, 0.0, 1.5))

    n = len(features)
    early = slice(0, max(1, n // 3))
    late = slice(max(1, (2 * n) // 3), n)
    dc_score = float(max(0.0, float(np.mean(arch[early]) - np.mean(arch))) + max(0.0, float(np.mean(hollow[late]) - np.mean(hollow))))
    feet_close_pct = float(np.mean(feet_hands < 1.2))
    muscleup_score = float(np.clip(0.55 * feet_close_pct + 0.45 * np.tanh(dc_score), 0.0, 1.5))

    scores = {
        "handstand": handstand_score,
        "swing360": swing_score,
        "muscleup": muscleup_score,
    }
    movement = max(scores, key=scores.get)
    best = scores[movement]

    if best < 0.3:
        return "unknown"
    if float(np.mean(inv)) > 0.55:
        return "handstand"
    return movement


def build_feedback(movement: str, features: list[FrameFeatures]) -> list[str]:
    if not features:
        return ["No se detectaron landmarks suficientes para dar feedback técnico."]

    elbow = np.array([f.elbow_bend_deg for f in features], dtype=np.float32)
    knee = np.array([f.knee_bend_deg for f in features], dtype=np.float32)
    stack = np.array([f.stack_error for f in features], dtype=np.float32)
    rot = np.array([f.rot_proxy for f in features], dtype=np.float32)
    shoulder_open = np.array([f.shoulder_open_deg for f in features], dtype=np.float32)
    hip_open = np.array([f.hip_open_deg for f in features], dtype=np.float32)
    feet_hands = np.array([f.feet_to_hands_norm for f in features], dtype=np.float32)

    if movement == "handstand":
        feedback: list[str] = []
        if float(np.mean(stack)) > 0.35:
            feedback.append("Alinea muneca-hombro-cadera-tobillo; microajusta con dedos para mejorar el balance.")
        if float(np.mean(elbow)) > 10.0:
            feedback.append("Bloquea codos y empuja el piso con hombros activos para ganar estabilidad.")
        if float(np.mean(knee)) > 10.0:
            feedback.append("Mantiene piernas extendidas y puntas activas para una linea mas limpia.")
        if float(np.mean(hip_open)) < 170.0:
            feedback.append("Reduce el arco lumbar: abdomen firme y ligera retroversion pelvica.")
        return feedback or ["Handstand estable: mantienes buena linea general y control del cuerpo."]

    if movement == "swing360":
        feedback = []
        if float(np.mean(rot)) < 0.02:
            feedback.append("Te falta velocidad de rotacion: coordina snap de hombros y cadera al despegar.")
        if float(np.mean(elbow)) > 25.0:
            feedback.append("Mantente mas compacto: codos cerca del cuerpo durante la fase aerea.")
        if float(np.mean(knee)) > 20.0:
            feedback.append("Evita abrir piernas; busca una patada mas limpia y controlada.")
        if float(np.mean(shoulder_open)) < 150.0:
            feedback.append("Mejora la posicion de hombros para transferir mejor impulso a la rotacion.")
        return feedback or ["Swing 360 consistente: buen ritmo de giro y control de postura."]

    if movement == "muscleup":
        feedback = []
        if float(np.mean(feet_hands < 1.2)) < 0.25:
            feedback.append("Sube mas rodillas/pies hacia la barra para cargar mejor el kip.")
        if float(np.mean(elbow)) > 25.0:
            feedback.append("Evita flexionar temprano los codos en el kip; conserva brazos largos al inicio.")
        if float(np.mean(hip_open)) < 160.0:
            feedback.append("Marca mejor la transicion D->C: extension inicial y cierre a hollow antes del tiron.")
        return feedback or ["Muscle-up bien encadenado: buena transferencia del kip a la fase de tiron."]

    return [
        "Movimiento no identificado con confianza.",
        "Intenta grabar con vista lateral y cuerpo completo para mejorar la deteccion.",
        "Mantiene una ejecucion clara y repetible para distinguir mejor el patron tecnico.",
    ]

