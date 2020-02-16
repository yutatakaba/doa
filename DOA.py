import argparse
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA  # linalgモジュールはLAとしてimportするのが慣例。
import math

parser = argparse.ArgumentParser(description="到来方向推定を行う")
parser.add_argument("--sensor", default=5, type=int, help="センサ数")
parser.add_argument("--sound_source", default=2, type=int, help="音源数")
parser.add_argument("--snap_shot", default=4, type=int, help="スナップショット数")
parser.add_argument("--mode", default="random", help="データの作成方法 sample: 例, random: 乱数生成")
parser.add_argument("--method", default="both", help="使用する手法 beam: BF方, music:MUSIC法 both: 両方")
parser.add_argument("--step", default=1000, type=int, help="推定時のステップ数")
parser.add_argument("--save", default="show", help="プロット画像を保存するかどうか save: 保存する, show: 見るだけ")
args = parser.parse_args()


class DOA:
    def __init__(self, M, J, n, mode, method, step):
        """
        到来方向推定を行うクラス
        Parameters
        ----------
        M: int
            センサ数
        J: int
            音源数
        n: int
            スナップショット数
        mode: str
            データの作成方法 sample: 例, random: 乱数生成
        method: str
            使用する手法 beam: BF方, music:MUSIC法
        step: int
            推定時のステップ数
        """
        self.M = M
        self.J = J
        self.n = n
        self.mode = mode
        self.method = method
        self.step = step
        self.e = np.e
        self.div = np.pi/step

    def make_h(self, dig):
        """
        ステアリング行列の作成を行う
        Parameters
        ----------
        dig: int
            角度(ラジアン)

        Returns
        -------
        H: array (type=complex)
            ステアリング行列
        """
        cos = math.cos(dig)
        _h = np.zeros(self.M, dtype=complex)
        for i in range(self.M):
            h_i = pow(self.e, (complex(0, math.sqrt(2)*2*i*np.pi*cos/4)))
            _h[i] = h_i

        H = _h.reshape(self.M, 1)
        return H

    def beam(self, R, ans_dig):
        """
        ビームフォーミング法を行う
        Parameters
        ----------
        R: array (type=complex)
            データ行列
        ans_dig: array
            真の到来方向

        Returns
        -------
        plot
            プロット画像

        See Also
        --------
        h_t: array (type=complex)
            hの共役転置
        BF_spec: array
            ビームフォーミングスペクトラグラム。到来方向のインデックスあたりにピークが立つ
        """
        start = 0
        BF_spec = np.zeros(self.step)

        """計算部分"""
        for x in range(self.step):
            h = self.make_h(start)
            h_t = np.conjugate(h.T)
            BF = np.dot(np.dot(h_t, R), h)
            BF_spec[x] = BF.real
            start += self.div

        """可視化処理"""
        x_axis = np.linspace(0, 180, self.step)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_axis, BF_spec)
        for dig in ans_dig:
            ax.axvline(dig, ls="--", color="black", alpha=0.3)
        plt.xlabel("Angle[°]")
        plt.show()
        if args.save == "save":
            plt.savefig("beam.png")

    def music(self, R, ans_dig):
        """
        MUSIC法を行う
        Parameters
        ----------
        R: array (type=complex)
            データ行列
        ans_dig: array
            真の到来方向

        Returns
        -------
        plot
            プロット画像

        See Also
        --------
        E_val: array (type=complex)
            Rの固有値
        E_vec: array (type=complex)
            Rの固有ベクトル
        E_big_id: index
            固有値の降順ソート結果
        _E_val: array (type=complex)
            Rの固有値を降順ソートした結果
        _E_vec: array (type=complex)
            Rの固有ベクトルを降順ソートした結果
        E, En: array (type=complex)
            雑音の作る空間
        h_t: array (type=complex)
            hの共役転置
        E_n_t: array (type=complex)
            Enの共役転置
        music_spec: array
            MUSICスペクトラグラム。到来方向のインデックスあたりにピークが立つ
        """
        E_val, E_vec = LA.eig(R)
        E_big_id = np.argsort(E_val)[::-1]
        _E_val = E_val[E_big_id]
        _E_vec = E_vec[:, E_big_id]
        E = _E_vec[:, self.M-1:]
        start = 0
        music_spec = np.zeros(self.step)

        """計算部分"""
        for x in range(self.step):
            h = self.make_h(start)
            h_t = np.conjugate(h.T)
            E_n = E
            E_n_t = np.conjugate(E_n.T)
            music_spec[x] = 1 / np.dot(np.dot(np.dot(h_t, E_n), E_n_t), h).real
            start += self.div

        """可視化処理"""
        x_axis = np.linspace(0, 180, self.step)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_axis, music_spec)
        for dig in ans_dig:
            ax.axvline(dig, ls="--", color="black", alpha=0.3)
        plt.xlabel("Angle[°]")
        plt.show()
        if args.save == "save":
            plt.savefig("music.png")


def gen_y(mode, _doa):
    """
    データ行列の作成を行う
    Parameters
    ----------
    mode: str
        データの作成方法 sample: 例, random: 乱数生成
    _doa: class object
        到来方向推定を行うクラス

    Returns
    -------
    _X: array (type=complex)
        データ行列
    _ans: array
        真の到来方向[度]

    See Also
    --------
    N_re: array
        実数部分のノイズ
    N_im: array
        虚数部分のノイズ
    N: array (type=complex)
        ノイズ
    H: array (type=complex)
        ステアリングベクトル
    S_re: array
        Sの実数部分
    S_im: array
        Sの虚数部分
    """
    if mode == 'sample':
        data = [
            [1.94-0.01j, 1.07+1.09j, -2.05-0.03j, -0.93+0.03j],
            [0.96+1.03j, 0.06-0.02j, -1.07-1.02j, -0.98-0.06j],
            [-0.01+0.04j, 0.97-1.05j, 0.08+0.04j, -1.06+0.03j]
        ]

        _X = np.array(data)
        _X = _X.T
        _ans = [90, 45]

        return _X, _ans

    elif mode == 'random':
        """ノイズの生成"""
        N_re = np.random.randint(-10, 10, (_doa.M, _doa.n))
        N_im = np.random.randint(-10, 10, (_doa.M, _doa.n))
        N = np.zeros((_doa.M, _doa.n), dtype=complex)
        for j in range(0, _doa.M):
            for k in range(0, _doa.n):
                N[_doa.J][k] = complex(N_re[j][k]/100, N_im[j][k]/100)
        _ans = np.random.randint(0, 180, _doa.J)
        H = []
        for dig in _ans:
            h = _doa.make_h(math.radians(dig))
            H.append(h)
        H = np.array(H, dtype=complex)

        """Sの生成"""
        S_re = np.random.randint(-10, 10, (_doa.J, _doa.n))
        S_im = np.random.randint(-10, 10, (_doa.J, _doa.n))
        S = np.zeros((_doa.J, _doa.n), dtype=complex)
        for j in range(0, _doa.J):
            for k in range(0, _doa.n):
                S[j][k] = complex(S_re[j][k], S_im[j][k])
        S = np.array(S, dtype=complex)
        _X = np.dot(H.T, S)
        _X = _X[0].T

        return _X, _ans


def make_R(M, n, x):
    """
    Rの作成を行う
    Parameters
    ----------
    M: int
            センサ数
    n: int
        スナップショット数
    x: array (type=complex)
        データ行列

    Returns
    -------
    R: array (type=complex)
        被固有値分解行列
    """
    R = [[0 for i in range(M)] for j in range(M)]
    for i in range(M):
        for j in range(M):
            R_ij = 0
            for k in range(n):
                R_ij += x[k, i] * np.conjugate(x[k, j])
            R[i][j] = R_ij/n
    R = np.array(R)

    return R


if __name__ == "__main__":
    if args.mode not in ['sample', 'random']:
        raise ValueError("'mode' must be 'sample' or 'random'")
    if args.method not in ['beam', 'music', 'both']:
        raise ValueError("'method' must be 'beam', 'music' or 'both'")
    if args.save not in ['show', 'save']:
        raise ValueError("'save' must be 'show' or 'save'")

    doa = DOA(args.sensor, args.sound_source, args.snap_shot, args.mode, args.method, args.step)
    X, ans = gen_y(args.mode, doa)
    _R = make_R(args.sensor, args.snap_shot, X)
    if args.method == 'beam':
        doa.beam(_R, ans)
    elif args.method == 'music':
        doa.music(_R, ans)
    elif args.method == 'both':
        doa.beam(_R, ans)
        doa.music(_R, ans)

