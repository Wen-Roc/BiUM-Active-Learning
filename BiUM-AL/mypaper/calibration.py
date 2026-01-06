# mypaper/calibration.py
import numpy as np
import math
from scipy.optimize import least_squares

# ==========================================================
#  欧拉角 ZYX → 旋转矩阵
# ==========================================================
def euler_to_rotmat(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]])
    return Rz @ Ry @ Rx

# ==========================================================
#  磁偶极子模型
# ==========================================================
def dipole_field_param(xl, yl, zl, a, b, c, m, n, p, BT):
    rx, ry, rz = xl - a, yl - b, zl - c
    R = np.sqrt(rx**2 + ry**2 + rz**2)
    mdotr = m * rx + n * ry + p * rz
    R3, R5 = R**3, R**5
    Blx = -BT * (3 * mdotr * rx / R5 - m / R3) * 1e6
    Bly = -BT * (3 * mdotr * ry / R5 - n / R3) * 1e6
    Blz = -BT * (3 * mdotr * rz / R5 - p / R3) * 1e6
    return Blx, Bly, Blz

# ==========================================================
#  残差函数
# ==========================================================
def residuals(params, xl, yl, zl, m, n, p, Bx_real, By_real, Bz_real):
    BT, a, b, c, ox, oy, oz, roll, pitch, yaw = params
    Bx_pred, By_pred, Bz_pred = dipole_field_param(xl, yl, zl, a, b, c, m, n, p, BT)
    B = np.vstack([Bx_pred + ox, By_pred + oy, Bz_pred + oz]).T
    R = euler_to_rotmat(roll, pitch, yaw)
    B_rot = (R @ B.T).T
    return np.concatenate([B_rot[:,0] - Bx_real, B_rot[:,1] - By_real, B_rot[:,2] - Bz_real])

# ==========================================================
#  预测函数
# ==========================================================
def predict_field(params, xl, yl, zl, m, n, p):
    BT, a, b, c, ox, oy, oz, roll, pitch, yaw = params
    Bx_pred, By_pred, Bz_pred = dipole_field_param(xl, yl, zl, a, b, c, m, n, p, BT)
    B = np.vstack([Bx_pred + ox, By_pred + oy, Bz_pred + oz]).T
    R = euler_to_rotmat(roll, pitch, yaw)
    B_rot = (R @ B.T).T
    return B_rot[:,0], B_rot[:,1], B_rot[:,2]

# ==========================================================
#  校准主函数
# ==========================================================
def calibrate_sensor(B_real, cc_params, sensor_loc, BT0=None, max_iter=800):
    """
    对单个传感器进行磁场校准
    Args:
        B_real: (N,3) 实测磁场数据
        cc_params: (N,6) cc 数据 [xl, yl, zl, m, n, p]
        sensor_loc: (3,) 传感器初始位置 [a,b,c]
        BT0: 初始 BT
        max_iter: LM 最大迭代次数
    Returns:
        params_opt: 校准后的参数
    """
    xl, yl, zl = cc_params[:,0], cc_params[:,1], cc_params[:,2]
    m, n, p = cc_params[:,3], cc_params[:,4], cc_params[:,5]

    if BT0 is None:
        # 默认 BT0
        d, l = 10, 10  # mm
        Br = 1.813519
        u0 = 4 * math.pi * 1e-7
        Ms = Br / u0
        magnet_volume = math.pi * (d / 2 / 1000) ** 2 * (l / 1000)
        mu = Ms * magnet_volume
        BT0 = (u0 * mu) / (4 * math.pi)

    a0, b0, c0 = sensor_loc
    ox0, oy0, oz0 = 0,0,0
    roll0, pitch0, yaw0 = 0,0,0
    params0 = np.array([BT0, a0, b0, c0, ox0, oy0, oz0, roll0, pitch0, yaw0])

    result = least_squares(
        residuals,
        params0,
        method='lm',
        max_nfev=max_iter,
        args=(xl, yl, zl, m, n, p, B_real[:,0], B_real[:,1], B_real[:,2])
    )
    return result.x

# ==========================================================
#  多传感器批量校准
# ==========================================================
def calibrate_all_sensors(B_all, cc_params, sensor_locs):
    """
    Args:
        B_all: (N,3*M) M 个传感器，每个 3 列 [Bx,By,Bz,...]
        cc_params: (N,6)
        sensor_locs: (M,3)
    Returns:
        list of optimized parameters, 每个元素长度 10
    """
    N_sensors = sensor_locs.shape[0]
    params_list = []
    for i in range(N_sensors):
        B_real = B_all[:, i*3:i*3+3]
        params_opt = calibrate_sensor(B_real, cc_params, sensor_locs[i])
        params_list.append(params_opt)
    return params_list
