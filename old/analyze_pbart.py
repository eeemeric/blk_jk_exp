import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from itertools import combinations
import statsmodels.api as sm

# Define logistic function outside the class
def logistic(x, L, k, x0):
    """Logistic function: L / (1 + exp(-k*(x-x0)))"""
    return L / (1 + np.exp(-k * (x - x0)))


class pBARTAnalyzer:
    def __init__(self):
        self.data_dir = "data"
        self.sessions = self.load_sessions()
    
    def load_sessions(self):
        """Load all session JSON files"""
        json_files = glob.glob(os.path.join(self.data_dir, "pbart_session_*.json"))
        sessions = []
        
        for file in json_files:
            with open(file, 'r') as f:
                session_data = json.load(f)
                sessions.append(session_data)
        
        return sessions
    
    def combine_all_trials(self):
        """Combine all trials from all sessions into single dataframe"""
        all_trials = []
        
        for session in self.sessions:
            for trial in session['trials']:
                all_trials.append(trial)
        
        df = pd.DataFrame(all_trials)
        return df
    
    def plot_consolidated_analysis(self, df):
        """Create consolidated figure with P(HIT), RT distribution, HIT vs STAY boxplot, and RT by tokens (2 granularities)"""
        
        # Calculate tokens_before_choice for each trial
        df_copy = df.copy()
        df_copy['tokens_before_choice'] = df_copy['sequence_total'] - df_copy['tokens_earned']
        
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 20))
        
        # ===== SUBPLOT 1: P(HIT) =====
        grouped = df_copy.groupby('tokens_before_choice').agg({
            'choice': lambda x: (x == 'HIT').sum() / len(x)
        }).reset_index()
        grouped.columns = ['tokens_before_choice', 'p_hit']
        grouped['n_trials'] = df_copy.groupby('tokens_before_choice').size().values
        
        ax1.scatter(grouped['tokens_before_choice'], grouped['p_hit'], 
                   s=grouped['n_trials']*30, alpha=0.6, color='steelblue')
        print(grouped)
        
        # try logistic fit
        if len(grouped) > 2:
            weights = grouped['n_trials'].values / grouped['n_trials'].max()
            try:
                x = np.linspace(0, 20, 21)
                # Fit Binomial GLM (Logistic Regression)
                X = sm.add_constant(x)                  # Add intercept
                model = sm.GLM(grouped['p_hit'], X, family=sm.families.Binomial())
                result = model.fit()
                # Predict and Plot
                x_smooth = np.linspace(0, 20, 100)
                X_smooth = sm.add_constant(x_smooth)
                y_pred = result.predict(X_smooth)
                ax1.plot(x_smooth, y_pred, "r--")
            except:
                pass
        
#         # Try polynomial fit
#         if len(grouped) > 2:
#             weights = grouped['n_trials'].values / grouped['n_trials'].max()
#             try:
#                 z = np.polyfit(grouped['tokens_before_choice'], grouped['p_hit'], 2, w=weights)
#                 p = np.poly1d(z)
#                 x_line = np.linspace(grouped['tokens_before_choice'].min(), 
#                                      grouped['tokens_before_choice'].max(), 100)
#                 ax1.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label='Polynomial fit')
#             except:
#                 pass
        
        ax1.set_xlabel('Tokens Earned at Start of Trial', fontsize=11)
        ax1.set_ylabel('Probability of HIT', fontsize=11)
        ax1.set_title('P(HIT) as a Function of Tokens at Trial Onset', fontsize=12, fontweight='bold')
        ax1.set_ylim(-0.2, 1.2)
        ax1.grid(True, alpha=0.3)
        ax1.set_box_aspect(0.62)
        ax1.legend()
        
        # ===== SUBPLOT 2: RT Histogram =====
        ax2.hist(df_copy['reaction_time_ms'], bins=30, color='steelblue', 
                edgecolor='black', alpha=0.5, density=True)
        
        mean_rt = df_copy['reaction_time_ms'].mean()
        median_rt = df_copy['reaction_time_ms'].median()
        ax2.axvline(mean_rt, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rt:.0f}ms')
        ax2.axvline(median_rt, color='green', linestyle='--', linewidth=2, label=f'Median: {median_rt:.0f}ms')
        
        ax2.set_xlabel('Reaction Time (ms)', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('Reaction Time Distribution: HIT and STAY', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 3000)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_box_aspect(0.62)
        # ===== SUBPLOT 3: HIT vs STAY Box Plot (Horizontal) =====
        hit_rts = df[df['choice'] == 'HIT']['reaction_time_ms']
        stay_rts = df[df['choice'] == 'STAY']['reaction_time_ms']
        
        bp = ax3.boxplot([hit_rts, stay_rts], positions=[1, 2], widths=0.5,
                         labels=['HIT', 'STAY'], patch_artist=True, vert=False)
        
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax3.scatter(hit_rts, [1]*len(hit_rts), alpha=0.2, s=15, color='blue')
        ax3.scatter(stay_rts, [2]*len(stay_rts), alpha=0.2, s=15, color='red')
        
        ax3.set_xlabel('Reaction Time (ms)', fontsize=11)
        ax3.set_ylabel('Choice Type', fontsize=11)
        ax3.set_title('HIT vs STAY Reaction Times', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 3000)
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.set_box_aspect(0.62)
        # ===== SUBPLOT 4: RT by Individual Token Levels (1-token bins) =====
        df_binned_1 = df_copy.copy()
        df_binned_1['token_bin'] = df_binned_1['tokens_before_choice'].astype(int).astype(str)
        
        rt_data_1 = [df_binned_1[df_binned_1['tokens_before_choice'] == i]['reaction_time_ms'].values 
                     for i in sorted(df_binned_1['tokens_before_choice'].unique())]
        bin_labels_1 = [str(int(i)) for i in sorted(df_binned_1['tokens_before_choice'].unique())]
        
        if len(rt_data_1) > 0:
            bp4 = ax4.boxplot(rt_data_1, labels=bin_labels_1, patch_artist=True, vert=False)
            
            colors_box_4 = plt.cm.viridis(np.linspace(0, 1, len(bin_labels_1)))
            for patch, color in zip(bp4['boxes'], colors_box_4):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Overlay individual points
            for i, rt_vals in enumerate(rt_data_1):
                y = np.random.normal(i + 1, 0.04, size=len(rt_vals))
                ax4.scatter(rt_vals, y, alpha=0.15, s=15, color='black')
            
            ax4.set_xlabel('Reaction Time (ms)', fontsize=11)
            ax4.set_ylabel('Tokens at Trial Onset', fontsize=11)
            ax4.set_title('Reaction Time Distribution by Individual Token Levels', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
            ax4.set_xlim(0, 3000)
            ax4.set_box_aspect(0.62)
        # ===== SUBPLOT 5: RT by 5-Token Bins (0-5, 6-10, 11-15, 16-20) =====
        df_binned_5 = df_copy.copy()
        
        # Create custom bins: 0-5, 6-10, 11-15, 16-20
        bins = [0, 6, 11, 16, 21]
        bin_labels_5 = ['0-5', '6-10', '11-15', '16-20']
        df_binned_5['token_bin'] = pd.cut(df_binned_5['tokens_before_choice'], 
                                          bins=bins, 
                                          labels=bin_labels_5, 
                                          right=False,
                                          include_lowest=True)
        
        df_binned_5 = df_binned_5.dropna(subset=['token_bin'])
        
        # Prepare data for box plot
        rt_data_5 = [df_binned_5[df_binned_5['token_bin'] == bin_label]['reaction_time_ms'].values 
                     for bin_label in bin_labels_5 if bin_label in df_binned_5['token_bin'].values]
        bin_labels_5_used = [bin_label for bin_label in bin_labels_5 if bin_label in df_binned_5['token_bin'].values]
        
        if len(rt_data_5) > 0:
            bp5 = ax5.boxplot(rt_data_5, labels=bin_labels_5_used, patch_artist=True, vert=False)
            
            colors_box_5 = plt.cm.viridis(np.linspace(0, 1, len(bin_labels_5_used)))
            for patch, color in zip(bp5['boxes'], colors_box_5):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Overlay individual points
            for i, (bin_label, rt_vals) in enumerate(zip(bin_labels_5_used, rt_data_5)):
                y = np.random.normal(i + 1, 0.04, size=len(rt_vals))
                ax5.scatter(rt_vals, y, alpha=0.15, s=15, color='black')
            
            ax5.set_xlabel('Reaction Time (ms)', fontsize=11)
            ax5.set_ylabel('Token Bins at Trial Onset', fontsize=11)
            ax5.set_title('Reaction Time Distribution by 5-Token Bins', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
            ax5.set_xlim(0, 3000)
            ax5.set_box_aspect(0.62)
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'consolidated_analysis.png'), dpi=300, bbox_inches='tight')
        print("Saved: consolidated_analysis.png")
        plt.close()
    
    def generate_summary_statistics(self, df):
        """Generate summary statistics with statistical tests"""
        
        output = []
        output.append("\n" + "="*60)
        output.append("SUMMARY STATISTICS")
        output.append("="*60)
        
        # Basic counts
        output.append(f"\nTotal trials: {len(df)}")
        output.append(f"Total sequences: {df['sequence_number'].max() + 1}")
        output.append(f"Total tokens accumulated: {df['total_accumulated'].iloc[-1]}")
        
        # ===== HIT vs STAY Analysis =====
        output.append("\n" + "-"*60)
        output.append("HIT vs STAY REACTION TIMES")
        output.append("-"*60)
        
        hit_rts = df[df['choice'] == 'HIT']['reaction_time_ms']
        stay_rts = df[df['choice'] == 'STAY']['reaction_time_ms']
        
        output.append(f"\nHIT trials: n={len(hit_rts)}, Mean RT={hit_rts.mean():.1f}ms (SD={hit_rts.std():.1f})")
        output.append(f"STAY trials: n={len(stay_rts)}, Mean RT={stay_rts.mean():.1f}ms (SD={stay_rts.std():.1f})")
        
        # Independent t-test
        t_stat, p_value = stats.ttest_ind(hit_rts, stay_rts)
        output.append(f"\nIndependent t-test:")
        output.append(f"  t({len(hit_rts) + len(stay_rts) - 2}) = {t_stat:.3f}, p = {p_value:.4f}")
        
        if p_value < 0.05:
            output.append(f"  ✓ SIGNIFICANT difference between HIT and STAY RTs (p < 0.05)")
        else:
            output.append(f"  ✗ No significant difference between HIT and STAY RTs (p >= 0.05)")
        
        # Effect size (Cohen's d)
        cohens_d = (hit_rts.mean() - stay_rts.mean()) / np.sqrt(((len(hit_rts)-1)*hit_rts.std()**2 + (len(stay_rts)-1)*stay_rts.std()**2) / (len(hit_rts) + len(stay_rts) - 2))
        output.append(f"  Cohen's d = {cohens_d:.3f}")
        
        # Mann-Whitney U test (non-parametric alternative)
        u_stat, p_value_mw = stats.mannwhitneyu(hit_rts, stay_rts)
        output.append(f"\nMann-Whitney U test (non-parametric):")
        output.append(f"  U = {u_stat:.1f}, p = {p_value_mw:.4f}")
        
        # ===== RT by Tokens at Trial Onset (BINNED) =====
        output.append("\n" + "-"*60)
        output.append("REACTION TIME BY TOKENS AT TRIAL ONSET (BINNED: 0-5, 6-10, 11-15, 16-20)")
        output.append("-"*60)
        
        df_binned = df.copy()
        df_binned['tokens_before_choice'] = df_binned['sequence_total'] - df_binned['tokens_earned']
        
        # Create custom bins: 0-5, 6-10, 11-15, 16-20
        bins = [0, 6, 11, 16, 21]
        bin_labels = ['0-5', '6-10', '11-15', '16-20']
        df_binned['token_bin'] = pd.cut(df_binned['tokens_before_choice'], 
                                        bins=bins, 
                                        labels=bin_labels, 
                                        right=False,
                                        include_lowest=True)
        df_binned = df_binned.dropna(subset=['token_bin'])
        
        grouped = df_binned.groupby('token_bin', observed=True)['reaction_time_ms'].agg(['mean', 'std', 'count']).reset_index()
        grouped.columns = ['token_bin', 'mean_rt', 'std_rt', 'n_trials']
        
        output.append("\nToken Bin    | N Trials | Mean RT (ms) | SD (ms)")
        output.append("-" * 50)
        for _, row in grouped.iterrows():
            output.append(f"{str(row['token_bin']):12s} | {int(row['n_trials']):8d} | {row['mean_rt']:12.1f} | {row['std_rt']:8.1f}")
        
        # ===== Kruskal-Wallis test on BINNED data =====
        output.append("\n" + "-"*60)
        output.append("KRUSKAL-WALLIS TEST: Reaction Time Across Token Bins")
        output.append("-"*60)
        
        # Create groups of RTs for each bin
        rt_groups = []
        bin_names_list = grouped['token_bin'].astype(str).values
        
        for bin_name in bin_names_list:
            group_rts = df_binned[df_binned['token_bin'] == bin_name]['reaction_time_ms'].values
            rt_groups.append(group_rts)
        
        # Perform Kruskal-Wallis test
        h_stat, p_value_kw = stats.kruskal(*rt_groups)
        
        output.append(f"\nKruskal-Wallis H-test:")
        output.append(f"  H = {h_stat:.3f}, p = {p_value_kw:.4f}")
        output.append(f"  Comparing {len(bin_names_list)} token bins")
        
        if p_value_kw < 0.05:
            output.append(f"  ✓ SIGNIFICANT difference in RTs across token bins (p < 0.05)")
            output.append(f"     Subjects respond differently depending on accumulated tokens")
        else:
            output.append(f"  ✗ No significant difference in RTs across token bins (p >= 0.05)")
        
        # Effect size (eta-squared approximation)
        n_total = len(df_binned)
        eta_squared = (h_stat - len(bin_names_list) + 1) / (n_total - len(bin_names_list))
        output.append(f"  η² = {eta_squared:.4f} (effect size)")
        
        # ===== Post-hoc Pairwise Comparisons (Mann-Whitney U with Bonferroni correction) =====
        output.append("\n" + "-"*60)
        output.append("POST-HOC PAIRWISE COMPARISONS (Mann-Whitney U with Bonferroni)")
        output.append("-"*60)
        
        # Calculate number of comparisons for Bonferroni correction
        n_comparisons = len(list(combinations(range(len(bin_names_list)), 2)))
        bonferroni_alpha = 0.05 / n_comparisons
        output.append(f"\nBonferroni-corrected alpha: {bonferroni_alpha:.6f}")
        output.append(f"(Original alpha = 0.05, Number of comparisons = {n_comparisons})\n")
        
        output.append("Token Bin Comparison    | U-statistic | p-value | Bonferroni Sig?")
        output.append("-" * 70)
        
        significant_comparisons = []
        
        for (i, j) in combinations(range(len(bin_names_list)), 2):
            bin_i = bin_names_list[i]
            bin_j = bin_names_list[j]
            
            u_stat, p_val = stats.mannwhitneyu(rt_groups[i], rt_groups[j])
            
            is_sig = "✓" if p_val < bonferroni_alpha else "✗"
            
            output.append(f"{str(bin_i):12s} vs {str(bin_j):12s} | {u_stat:11.1f} | {p_val:.6f} | {is_sig}")
            
            if p_val < bonferroni_alpha:
                significant_comparisons.append((bin_i, bin_j, p_val))
        
        if significant_comparisons:
            output.append(f"\nSignificant pairwise differences (after Bonferroni correction):")
            for b1, b2, p in significant_comparisons:
                mean_rt_b1 = df_binned[df_binned['token_bin'] == b1]['reaction_time_ms'].mean()
                mean_rt_b2 = df_binned[df_binned['token_bin'] == b2]['reaction_time_ms'].mean()
                direction = "faster" if mean_rt_b1 < mean_rt_b2 else "slower"
                output.append(f"  • {b1} tokens: RT {direction} than {b2} tokens ({mean_rt_b1:.0f} vs {mean_rt_b2:.0f} ms, p={p:.6f})")
        else:
            output.append(f"\nNo pairwise differences survive Bonferroni correction.")
        
        output.append("\n" + "="*60 + "\n")
        
        # Save to file
        output_text = "\n".join(output)
        print(output_text)
        
        with open(os.path.join(self.data_dir, 'summary_statistics.txt'), 'w') as f:
            f.write(output_text)
        print("Saved: summary_statistics.txt")
    
    def run_analysis(self):
        """Run complete analysis"""
        if len(self.sessions) == 0:
            print("No session data found!")
            return
        
        df = self.combine_all_trials()
        
        print("\nGenerating analysis...")
        self.plot_consolidated_analysis(df)
        self.generate_summary_statistics(df)
        
        print("\nAnalysis complete!")


if __name__ == "__main__":
    analyzer = pBARTAnalyzer()
    analyzer.run_analysis()