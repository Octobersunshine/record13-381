import warnings
import numpy as np
from scipy import stats
from typing import Tuple, Union, List


class MannWhitneyUTest:
    def __init__(self, alternative: str = "two-sided", use_continuity: bool = True):
        if alternative not in ("two-sided", "less", "greater"):
            raise ValueError(
                "alternative must be 'two-sided', 'less', or 'greater'"
            )
        self.alternative = alternative
        self.use_continuity = use_continuity

    @staticmethod
    def _choose_method(n1: int, n2: int) -> str:
        min_n = min(n1, n2)
        max_n = max(n1, n2)
        ratio = max_n / min_n if min_n > 0 else float("inf")
        combined = n1 + n2

        if min_n <= 8 and combined <= 50:
            return "exact"

        if ratio > 3.0 and min_n <= 20:
            return "exact"

        if min_n <= 3:
            return "exact"

        return "asymptotic"

    def _validate_samples(self, x: np.ndarray, y: np.ndarray):
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

    def test(
        self,
        sample1: Union[List[float], np.ndarray],
        sample2: Union[List[float], np.ndarray],
        method: str = "auto",
    ) -> Tuple[float, float, dict]:
        if method not in ("auto", "exact", "asymptotic"):
            raise ValueError("method must be 'auto', 'exact', or 'asymptotic'")

        x = np.asarray(sample1, dtype=np.float64)
        y = np.asarray(sample2, dtype=np.float64)
        self._validate_samples(x, y)

        n1 = x.size
        n2 = y.size

        if method == "auto":
            resolved_method = self._choose_method(n1, n2)
        else:
            resolved_method = method

        if resolved_method == "exact":
            scipy_method = "exact"
        else:
            scipy_method = "asymptotic"

        if resolved_method == "asymptotic":
            min_n = min(n1, n2)
            max_n = max(n1, n2)
            ratio = max_n / min_n if min_n > 0 else float("inf")
            if ratio > 3.0:
                warnings.warn(
                    f"Sample size ratio ({max_n}/{min_n} = {ratio:.1f}) is large. "
                    f"Asymptotic p-value may be inaccurate. "
                    f"Consider using method='exact' or method='auto'.",
                    UserWarning,
                    stacklevel=2,
                )

        result = stats.mannwhitneyu(
            x,
            y,
            alternative=self.alternative,
            method=scipy_method,
        )

        u_statistic = float(result.statistic)
        p_value = float(result.pvalue)

        u_max = n1 * n2
        u1 = u_statistic
        u2 = u_max - u1
        median_diff = float(np.median(x) - np.median(y))

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
            "median_difference": median_diff,
            "median_sample1": float(np.median(x)),
            "median_sample2": float(np.median(y)),
            "mean_rank_sample1": float(self._mean_rank(n1, u1)),
            "mean_rank_sample2": float(self._mean_rank(n2, u2)),
            "alternative": self.alternative,
            "method": resolved_method,
            "significance_level": 0.05,
            "conclusion": conclusion,
        }

        if resolved_method == "asymptotic" and method != "asymptotic":
            try:
                exact_result = stats.mannwhitneyu(
                    x, y, alternative=self.alternative, method="exact"
                )
                info["p_value_exact"] = float(exact_result.pvalue)
                info["p_value_asymptotic"] = p_value
                info["p_value_diff"] = abs(p_value - float(exact_result.pvalue))
            except Exception:
                pass

        return u_statistic, p_value, info

    @staticmethod
    def _mean_rank(n_a: int, u_val: float) -> float:
        if n_a == 0:
            return 0.0
        return u_val / n_a + (n_a + 1) / 2.0


def mann_whitney_u_test(
    sample1: Union[List[float], np.ndarray],
    sample2: Union[List[float], np.ndarray],
    alternative: str = "two-sided",
    use_continuity: bool = True,
    method: str = "auto",
) -> Tuple[float, float, dict]:
    tester = MannWhitneyUTest(
        alternative=alternative, use_continuity=use_continuity
    )
    return tester.test(sample1, sample2, method=method)


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("场景1: 样本量相近 (30 vs 25) — 渐近法通常可靠")
    print("=" * 70)
    sample_a = np.random.exponential(scale=2.0, size=30)
    sample_b = np.random.exponential(scale=3.0, size=25)

    u, p, info = mann_whitney_u_test(
        sample_a, sample_b, method="auto"
    )
    print(f"  样本量: n1={info['n1']}, n2={info['n2']}, 比值={max(info['n1'],info['n2'])/min(info['n1'],info['n2']):.1f}")
    print(f"  自动选择方法: {info['method']}")
    print(f"  U 统计量: {u:.4f}")
    print(f"  p 值: {p:.6f}")
    print(f"  结论: {info['conclusion']}")

    print("\n" + "=" * 70)
    print("场景2: 样本量差异过大 (5 vs 200) — 渐近法 p 值不准确 (Bug 演示)")
    print("=" * 70)
    small = [1.2, 2.5, 3.1, 1.8, 2.2]
    large = np.random.exponential(scale=2.0, size=200).tolist()

    print("\n  [渐近法 - 存在偏差]")
    u_asymp, p_asymp, info_asymp = mann_whitney_u_test(
        small, large, method="asymptotic"
    )
    print(f"  U 统计量: {u_asymp:.4f}")
    print(f"  p 值 (渐近): {p_asymp:.6f}")

    print("\n  [精确法 - 准确结果]")
    u_exact, p_exact, info_exact = mann_whitney_u_test(
        small, large, method="exact"
    )
    print(f"  U 统计量: {u_exact:.4f}")
    print(f"  p 值 (精确): {p_exact:.6f}")
    print(f"  p 值差异: {abs(p_asymp - p_exact):.6f}")
    if abs(p_asymp - p_exact) > 0.01:
        print(f"  ⚠ 渐近法与精确法 p 值差异超过 0.01，渐近法不可靠！")

    print("\n  [自动模式 - 智能选择]")
    u_auto, p_auto, info_auto = mann_whitney_u_test(
        small, large, method="auto"
    )
    print(f"  自动选择方法: {info_auto['method']}")
    print(f"  U 统计量: {u_auto:.4f}")
    print(f"  p 值: {p_auto:.6f}")

    print("\n" + "=" * 70)
    print("场景3: 样本量差异过大 (8 vs 100) — 自动模式警告演示")
    print("=" * 70)
    med = np.random.exponential(scale=2.0, size=8).tolist()
    huge = np.random.exponential(scale=2.0, size=100).tolist()

    u_auto2, p_auto2, info_auto2 = mann_whitney_u_test(
        med, huge, method="auto"
    )
    print(f"  样本量: n1=8, n2=100, 比值=12.5")
    print(f"  自动选择方法: {info_auto2['method']}")
    print(f"  U 统计量: {u_auto2:.4f}")
    print(f"  p 值: {p_auto2:.6f}")

    print("\n" + "=" * 70)
    print("场景4: 强制渐近法 + 大比值 — 触发用户警告")
    print("=" * 70)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        u_warn, p_warn, info_warn = mann_whitney_u_test(
            med, huge, method="asymptotic"
        )
        print(f"  U 统计量: {u_warn:.4f}")
        print(f"  p 值 (渐近): {p_warn:.6f}")
        if w:
            print(f"  ⚠ 警告: {w[0].message}")
        if "p_value_exact" in info_warn:
            print(f"  p 值 (精确参考): {info_warn['p_value_exact']:.6f}")
            print(f"  p 值差异: {info_warn['p_value_diff']:.6f}")
