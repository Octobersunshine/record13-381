import numpy as np
from scipy import stats
from typing import Tuple, Optional, Union, List


class MannWhitneyUTest:
    def __init__(self, alternative: str = "two-sided", use_continuity: bool = True):
        if alternative not in ("two-sided", "less", "greater"):
            raise ValueError(
                "alternative must be 'two-sided', 'less', or 'greater'"
            )
        self.alternative = alternative
        self.use_continuity = use_continuity

    def test(
        self,
        sample1: Union[List[float], np.ndarray],
        sample2: Union[List[float], np.ndarray],
    ) -> Tuple[float, float, dict]:
        x = np.asarray(sample1, dtype=np.float64)
        y = np.asarray(sample2, dtype=np.float64)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("samples must be 1-D arrays")
        if x.size < 1 or y.size < 1:
            raise ValueError("samples must not be empty")
        if x.size == 1 and y.size == 1:
            raise ValueError(
                "at least one sample must have more than one observation"
            )
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("samples must not contain NaN or infinite values")

        result = stats.mannwhitneyu(
            x,
            y,
            alternative=self.alternative,
            method="asymptotic",
        )

        u_statistic = float(result.statistic)
        p_value = float(result.pvalue)

        n1 = x.size
        n2 = y.size
        u_max = n1 * n2
        u1 = u_statistic
        u2 = u_max - u1
        median_diff = np.median(x) - np.median(y)

        if self.alternative == "two-sided":
            conclusion = (
                "Reject H0: distributions are significantly different"
                if p_value < 0.05
                else "Fail to reject H0: no significant difference"
            )
        elif self.alternative == "less":
            conclusion = (
                "Reject H0: sample1 distribution is significantly less than sample2"
                if p_value < 0.05
                else "Fail to reject H0"
            )
        else:
            conclusion = (
                "Reject H0: sample1 distribution is significantly greater than sample2"
                if p_value < 0.05
                else "Fail to reject H0"
            )

        info = {
            "n1": n1,
            "n2": n2,
            "u1": u1,
            "u2": u2,
            "median_difference": float(median_diff),
            "median_sample1": float(np.median(x)),
            "median_sample2": float(np.median(y)),
            "mean_rank_sample1": float(self._mean_rank(x, y, u1)),
            "mean_rank_sample2": float(self._mean_rank(x, y, u2)),
            "alternative": self.alternative,
            "significance_level": 0.05,
            "conclusion": conclusion,
        }

        return u_statistic, p_value, info

    def test_exact(
        self,
        sample1: Union[List[float], np.ndarray],
        sample2: Union[List[float], np.ndarray],
    ) -> Tuple[float, float, dict]:
        x = np.asarray(sample1, dtype=np.float64)
        y = np.asarray(sample2, dtype=np.float64)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("samples must be 1-D arrays")
        if x.size < 1 or y.size < 1:
            raise ValueError("samples must not be empty")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("samples must not contain NaN or infinite values")

        result = stats.mannwhitneyu(
            x,
            y,
            alternative=self.alternative,
            method="exact",
        )

        u_statistic = float(result.statistic)
        p_value = float(result.pvalue)
        info = {"method": "exact", "alternative": self.alternative}

        return u_statistic, p_value, info

    @staticmethod
    def _mean_rank(sample_a: np.ndarray, sample_b: np.ndarray, u_val: float) -> float:
        n_a = sample_a.size
        if n_a == 0:
            return 0.0
        n_b = sample_b.size
        total_n = n_a + n_b
        mean_rank = u_val / n_a + (n_a + 1) / 2.0
        return mean_rank


def mann_whitney_u_test(
    sample1: Union[List[float], np.ndarray],
    sample2: Union[List[float], np.ndarray],
    alternative: str = "two-sided",
    use_continuity: bool = True,
    exact: bool = False,
) -> Tuple[float, float, dict]:
    tester = MannWhitneyUTest(
        alternative=alternative, use_continuity=use_continuity
    )
    if exact:
        return tester.test_exact(sample1, sample2)
    return tester.test(sample1, sample2)


if __name__ == "__main__":
    np.random.seed(42)

    sample_a = np.random.exponential(scale=2.0, size=30)
    sample_b = np.random.exponential(scale=3.0, size=25)

    print("=" * 60)
    print("Mann-Whitney U 检验示例")
    print("=" * 60)
    print(f"\n样本 A 大小: {sample_a.size}, 中位数: {np.median(sample_a):.4f}")
    print(f"样本 B 大小: {sample_b.size}, 中位数: {np.median(sample_b):.4f}")

    u, p, info = mann_whitney_u_test(
        sample_a, sample_b, alternative="two-sided"
    )

    print(f"\n--- 检验结果 ---")
    print(f"U 统计量: {u:.4f}")
    print(f"p 值: {p:.6f}")
    print(f"U1 (A vs B): {info['u1']:.4f}")
    print(f"U2 (B vs A): {info['u2']:.4f}")
    print(f"中位数差 (A - B): {info['median_difference']:.4f}")
    print(f"平均秩 (A): {info['mean_rank_sample1']:.4f}")
    print(f"平均秩 (B): {info['mean_rank_sample2']:.4f}")
    print(f"备择假设: {info['alternative']}")
    print(f"结论 (α=0.05): {info['conclusion']}")

    print("\n" + "=" * 60)
    print("单侧检验示例: 样本 A < 样本 B")
    print("=" * 60)
    u_less, p_less, info_less = mann_whitney_u_test(
        sample_a, sample_b, alternative="less"
    )
    print(f"U 统计量: {u_less:.4f}")
    print(f"p 值: {p_less:.6f}")
    print(f"结论: {info_less['conclusion']}")

    print("\n" + "=" * 60)
    print("小样本精确检验示例")
    print("=" * 60)
    small_a = [1.2, 2.5, 3.1, 1.8, 2.2]
    small_b = [3.5, 4.2, 5.1, 4.8, 3.9, 5.5]
    u_exact, p_exact, info_exact = mann_whitney_u_test(
        small_a, small_b, exact=True
    )
    print(f"样本 A: {small_a}")
    print(f"样本 B: {small_b}")
    print(f"U 统计量 (精确法): {u_exact:.4f}")
    print(f"p 值 (精确法): {p_exact:.6f}")
