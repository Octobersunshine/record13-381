import warnings
import numpy as np
from scipy import stats
from typing import Tuple, Union, List, Optional


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


class WilcoxonSignedRankTest:
    def __init__(self, alternative: str = "two-sided", zero_method: str = "wilcox"):
        if alternative not in ("two-sided", "less", "greater"):
            raise ValueError(
                "alternative must be 'two-sided', 'less', or 'greater'"
            )
        if zero_method not in ("wilcox", "pratt", "zsplit"):
            raise ValueError(
                "zero_method must be 'wilcox', 'pratt', or 'zsplit'"
            )
        self.alternative = alternative
        self.zero_method = zero_method

    @staticmethod
    def _choose_method(n: int, has_ties: bool) -> str:
        if n <= 50 and not has_ties:
            return "exact"
        return "asymptotic"

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

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("samples must be 1-D arrays")
        if x.size != y.size:
            raise ValueError("paired samples must have the same length")
        if x.size < 6:
            raise ValueError("sample size must be at least 6")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("samples must not contain NaN or infinite values")

        diff = x - y
        nonzero = diff[diff != 0]
        n_nonzero = nonzero.size
        has_ties = len(np.unique(np.abs(nonzero))) < n_nonzero

        if method == "auto":
            resolved_method = self._choose_method(n_nonzero, has_ties)
        else:
            resolved_method = method

        if resolved_method == "exact" and has_ties:
            warnings.warn(
                "Exact p-value may be unreliable when there are ties in the "
                "absolute differences. Consider using method='asymptotic' or "
                "use the 'pratt' zero_method.",
                UserWarning,
                stacklevel=2,
            )

        try:
            result = stats.wilcoxon(
                x,
                y,
                alternative=self.alternative,
                zero_method=self.zero_method,
                method=resolved_method if resolved_method != "asymptotic" else "approx",
            )
        except ValueError:
            result = stats.wilcoxon(
                x,
                y,
                alternative=self.alternative,
                zero_method=self.zero_method,
                method="approx",
            )
            resolved_method = "asymptotic"

        w_statistic = float(result.statistic)
        p_value = float(result.pvalue)

        median_diff = float(np.median(diff))
        n_pairs = x.size
        n_zeros = int(np.sum(diff == 0))
        n_positive = int(np.sum(diff > 0))
        n_negative = int(np.sum(diff < 0))

        if self.alternative == "two-sided":
            conclusion = (
                "Reject H0: paired samples have significantly different distributions"
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
            "n_pairs": n_pairs,
            "n_nonzero": n_nonzero,
            "n_zeros": n_zeros,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "median_difference": median_diff,
            "mean_difference": float(np.mean(diff)),
            "zero_method": self.zero_method,
            "alternative": self.alternative,
            "method": resolved_method,
            "significance_level": 0.05,
            "conclusion": conclusion,
        }

        return w_statistic, p_value, info


class KruskalWallisTest:
    def __init__(self, nan_policy: str = "raise"):
        if nan_policy not in ("propagate", "raise", "omit"):
            raise ValueError(
                "nan_policy must be 'propagate', 'raise', or 'omit'"
            )
        self.nan_policy = nan_policy

    def _validate_samples(self, samples: List[np.ndarray]):
        if len(samples) < 2:
            raise ValueError("at least 2 groups are required")
        for i, s in enumerate(samples):
            if s.ndim != 1:
                raise ValueError(f"sample {i+1} must be a 1-D array")
            if s.size < 1:
                raise ValueError(f"sample {i+1} must not be empty")

    def test(
        self,
        *samples: Union[List[float], np.ndarray],
    ) -> Tuple[float, float, dict]:
        arr_list = [np.asarray(s, dtype=np.float64) for s in samples]
        self._validate_samples(arr_list)

        result = stats.kruskal(
            *arr_list,
            nan_policy=self.nan_policy,
        )

        h_statistic = float(result.statistic)
        p_value = float(result.pvalue)

        k = len(arr_list)
        n_total = sum(arr.size for arr in arr_list)
        group_sizes = [arr.size for arr in arr_list]
        group_medians = [float(np.median(arr)) for arr in arr_list]

        all_data = np.concatenate(arr_list)
        ranks = stats.rankdata(all_data)
        group_ranks = []
        start = 0
        for size in group_sizes:
            group_ranks.append(float(np.mean(ranks[start:start + size])))
            start += size

        has_ties = len(np.unique(all_data)) < n_total

        h_adj = None
        if has_ties and n_total > 0:
            tied_values, counts = np.unique(all_data, return_counts=True)
            tie_correction = 1 - np.sum(counts**3 - counts) / (n_total**3 - n_total)
            if tie_correction > 0:
                h_adj = h_statistic / tie_correction

        df = k - 1

        conclusion = (
            "Reject H0: at least one group has a significantly different distribution"
            if p_value < 0.05
            else "Fail to reject H0: no significant difference among groups"
        )

        info = {
            "k_groups": k,
            "n_total": n_total,
            "group_sizes": group_sizes,
            "group_medians": group_medians,
            "group_mean_ranks": group_ranks,
            "degrees_of_freedom": df,
            "has_ties": has_ties,
            "h_corrected": h_adj,
            "tie_correction_applied": h_adj is not None,
            "nan_policy": self.nan_policy,
            "significance_level": 0.05,
            "conclusion": conclusion,
        }

        return h_statistic, p_value, info


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


def wilcoxon_signed_rank_test(
    sample1: Union[List[float], np.ndarray],
    sample2: Union[List[float], np.ndarray],
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
    method: str = "auto",
) -> Tuple[float, float, dict]:
    tester = WilcoxonSignedRankTest(
        alternative=alternative, zero_method=zero_method
    )
    return tester.test(sample1, sample2, method=method)


def kruskal_wallis_test(
    *samples: Union[List[float], np.ndarray],
    nan_policy: str = "raise",
) -> Tuple[float, float, dict]:
    tester = KruskalWallisTest(nan_policy=nan_policy)
    return tester.test(*samples)


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("【一】Mann-Whitney U 检验 (两独立样本)")
    print("=" * 70)
    sample_a = np.random.exponential(scale=2.0, size=30)
    sample_b = np.random.exponential(scale=3.0, size=25)

    u, p, info = mann_whitney_u_test(sample_a, sample_b, method="auto")
    print(f"  样本量: n1={info['n1']}, n2={info['n2']}")
    print(f"  方法: {info['method']}")
    print(f"  U 统计量: {u:.4f}")
    print(f"  p 值: {p:.6f}")
    print(f"  结论: {info['conclusion']}")

    print("\n" + "=" * 70)
    print("【二】Wilcoxon 符号秩检验 (配对样本)")
    print("=" * 70)
    n_pairs = 30
    pre = np.random.exponential(scale=2.5, size=n_pairs)
    post = pre + np.random.normal(loc=-0.5, scale=1.0, size=n_pairs)

    w, p_w, info_w = wilcoxon_signed_rank_test(pre, post, method="auto")
    print(f"  配对数: {info_w['n_pairs']} (非零差: {info_w['n_nonzero']})")
    print(f"  正差数: {info_w['n_positive']}, 负差数: {info_w['n_negative']}")
    print(f"  方法: {info_w['method']}")
    print(f"  W 统计量: {w:.4f}")
    print(f"  p 值: {p_w:.6f}")
    print(f"  中位数差 (前-后): {info_w['median_difference']:.4f}")
    print(f"  结论: {info_w['conclusion']}")

    print("\n  --- 单侧检验: post < pre (降低) ---")
    w_less, p_less, info_less = wilcoxon_signed_rank_test(
        pre, post, alternative="greater", method="auto"
    )
    print(f"  W 统计量: {w_less:.4f}")
    print(f"  p 值 (单侧 greater): {p_less:.6f}")
    print(f"  结论: {info_less['conclusion']}")

    print("\n" + "=" * 70)
    print("【三】Kruskal-Wallis H 检验 (多组独立样本)")
    print("=" * 70)
    group1 = np.random.exponential(scale=2.0, size=20)
    group2 = np.random.exponential(scale=3.0, size=25)
    group3 = np.random.exponential(scale=2.5, size=22)

    h, p_h, info_h = kruskal_wallis_test(group1, group2, group3)
    print(f"  组数: {info_h['k_groups']}, 总样本: {info_h['n_total']}")
    print(f"  各组样本量: {info_h['group_sizes']}")
    print(f"  各组中位数: {[round(m, 4) for m in info_h['group_medians']]}")
    print(f"  各组平均秩: {[round(r, 4) for r in info_h['group_mean_ranks']]}")
    print(f"  自由度: {info_h['degrees_of_freedom']}")
    print(f"  是否有结点: {info_h['has_ties']}")
    print(f"  H 统计量: {h:.4f}")
    if info_h['h_corrected'] is not None:
        print(f"  H (校正后): {info_h['h_corrected']:.4f}")
    print(f"  p 值: {p_h:.6f}")
    print(f"  结论: {info_h['conclusion']}")

    print("\n  --- 两组对比 (等价于 Mann-Whitney U 双侧检验) ---")
    h2, p2, info2 = kruskal_wallis_test(group1, group2)
    print(f"  两组比较 H 统计量: {h2:.4f}")
    print(f"  两组比较 p 值: {p2:.6f}")
    print(f"  自由度: {info2['degrees_of_freedom']}")

    print("\n" + "=" * 70)
    print("【四】三种检验关系总结")
    print("=" * 70)
    print("  场景                    | 推荐检验方法")
    print("  ------------------------|---------------------------")
    print("  两独立样本              | Mann-Whitney U 检验")
    print("  两配对/相关样本         | Wilcoxon 符号秩检验")
    print("  三组及以上独立样本      | Kruskal-Wallis H 检验")
    print("  (Kruskal-Wallis 两组时 ≈ Mann-Whitney U 双侧)")
