# mnkybx_pBART_analysis.py
# erik.emeric@gmail.com
# eemeric2@jhu.edu

import sqlite3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from itertools import combinations
import statsmodels.api as sm
from scipy.optimize import minimize
import numpy as np

# TODO: SORT BY RISK OF DECK (HIGH/MEDIUM/LOW)
# TODO: PROSPECT THEORY ANALYSIS
# Define logistic function outside the class
def logistic(x, L, k, x0):
    """Logistic function: L / (1 + exp(-k*(x-x0)))"""
    return L / (1 + np.exp(-k * (x - x0)))


class pBARTAnalyzer:
    def __init__(self):
        self.data_dir = "data"
        self.db_path = os.path.join(self.data_dir, "pbart_data.db")
        self.sessions = self.load_sessions_from_sqlite()

    def load_sessions_from_sqlite(self):
        """Load all sessions from SQLite database"""
        if not os.path.exists(self.db_path):
            print(f"Database not found at {self.db_path}")
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all sessions
        cursor.execute('SELECT session_id, subject_id, session_timestamp, total_tokens_accumulated FROM sessions')
        sessions = cursor.fetchall()

        session_list = []
        for session_id, subject_id, timestamp, total_tokens in sessions:
            # Get trials for this session
            cursor.execute('''
                SELECT * FROM trials WHERE session_id = ?
                ORDER BY trial_id
            ''', (session_id,))

            trials = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]

            session_data = {
                'session_id': session_id,
                'subject_id': subject_id,
                'session_timestamp': timestamp,
                'total_tokens_accumulated': total_tokens,
                'trials': [dict(zip(column_names, trial)) for trial in trials]
            }
            session_list.append(session_data)

        conn.close()
        return session_list


    def combine_all_trials(self):
        """Combine all trials from all sessions into single dataframe"""
        all_trials = []

        for session in self.sessions:
            for trial in session['trials']:
                all_trials.append(trial)

        df = pd.DataFrame(all_trials)
        return df


    def get_session_trials(self, session_index):
        """Get trials for a specific session"""
        session = self.sessions[session_index]
        df = pd.DataFrame(session['trials'])
        return df


    def get_binomial_fit(self, xdata, ydata):
        """
        Fit logistic regression to data.

        Args:
            xdata: Array of x values (e.g., points accumulated)
            ydata: Array of y values (e.g., P(HIT))

        Returns:
            dict with keys: 'model', 'x_smooth', 'y_pred', 'success'
        """
        result = {
            'model': None,
            'x_smooth': None,
            'y_pred': None,
            'success': False
        }

        try:
            if len(xdata) < 3:
                return result

            # Fit logistic regression
            X = sm.add_constant(xdata)
            model = sm.GLM(ydata, X, family=sm.families.Binomial())
            fitted_model = model.fit(disp=0)

            # Predict on smooth x
            x_smooth = np.linspace(xdata.min(), xdata.max(), 100)
            X_smooth = sm.add_constant(x_smooth)
            y_pred = fitted_model.predict(X_smooth)

            result['model'] = fitted_model
            result['x_smooth'] = x_smooth
            result['y_pred'] = y_pred
            result['success'] = True

        except Exception as e:
            print(f"Fit failed: {str(e)}")

        return result


    def get_indifference_point(self, model):
        """
        Calculate indifference point from fitted model (where y = 0.5).

        Args:
            model: Fitted statsmodels GLM model

        Returns:
            float: x-value where y=0.5, or None if calculation fails
        """
        try:
            params = model.params
            intercept = params[0]  # First param is always intercept (const)
            slope = params[1]  # Second param is always slope

            print(f"DEBUG: intercept={intercept}, slope={slope}")

            if slope != 0:
                ip = -intercept / slope
                print(f"DEBUG: indifference_point={ip}")
                return ip
            else:
                print("DEBUG: slope is 0")
        except Exception as e:
            print(f"Indifference point calculation failed: {str(e)}")

        return None


    def calculate_adjusted_average_pumps(self, df):
        """
        Calculate Adjusted Average Pumps (Standard Index) for a session/subject.
        This is the sum of hits on all WIN sequences divided by the total number of WIN sequences.

        A sequence is WIN if it ends with result != 'BUST'
        A sequence is BUST if the last trial has result == 'BUST'
        """
        if len(df) == 0:
            return 0, 0

        # Group by sequence_number and get the last trial of each sequence
        sequence_outcomes = {}

        for seq_num in df['sequence_number'].unique():
            seq_data = df[df['sequence_number'] == seq_num]
            last_trial = seq_data.iloc[-1]

            # The result of the sequence is determined by the last trial's result
            result = last_trial['result']

            # If result is 'BUST', it's a BUST sequence; otherwise it's a WIN
            if result == 'BUST':
                sequence_outcomes[seq_num] = 'BUST'
            else:
                sequence_outcomes[seq_num] = 'WIN'

        # Count WIN sequences
        n_win_sequences = sum(1 for outcome in sequence_outcomes.values() if outcome == 'WIN')

        if n_win_sequences == 0:
            return 0, 0

        # Count total HITs in WIN sequences only
        n_hits_in_wins = 0
        for seq_num, outcome in sequence_outcomes.items():
            if outcome == 'WIN':
                seq_data = df[df['sequence_number'] == seq_num]
                n_hits_in_wins += (seq_data['choice'] == 'HIT').sum()

        # Adjusted Average Pumps = total hits in WIN sequences / total WIN sequences
        adjusted_avg_pumps = n_hits_in_wins / n_win_sequences

        return adjusted_avg_pumps, n_win_sequences


    def plot_consolidated_analysis(self, df, filename="consolidated_analysis.png"):
        """Create consolidated figure with P(HIT), RT distribution, HIT vs STAY boxplot, and RT by tokens (2 granularities)"""

        # Calculate tokens_before_choice for each trial
        df_copy = df.copy()
        df_copy['tokens_before_choice'] = df_copy['sequence_total'] - df_copy['tokens_this_hit']

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 20))

        # ===== SUBPLOT 1: P(HIT) =====
        grouped = df_copy.groupby('tokens_before_choice').agg({
            'choice': lambda x: (x == 'HIT').sum() / len(x)
        }).reset_index()
        grouped.columns = ['tokens_before_choice', 'p_hit']
        grouped['n_trials'] = df_copy.groupby('tokens_before_choice').size().values

        ax1.scatter(grouped['tokens_before_choice'], grouped['p_hit'],
                    s=grouped['n_trials'] * 30, alpha=0.6, color='steelblue')

        # try logistic fit
        if len(grouped) > 2:
            weights = grouped['n_trials'].values / grouped['n_trials'].max()
            try:
                # Use ACTUAL grouped data, not artificial x values
                X = sm.add_constant(grouped['tokens_before_choice'])  # Use real x values
                model = sm.GLM(grouped['p_hit'], X, family=sm.families.Binomial())
                result = model.fit()

                # Predict on smooth x for plotting
                x_smooth = np.linspace(grouped['tokens_before_choice'].min(),
                                       grouped['tokens_before_choice'].max(), 100)
                X_smooth = sm.add_constant(x_smooth)
                y_pred = result.predict(X_smooth)
                ax1.plot(x_smooth, y_pred, "r--", linewidth=2, label='Logistic fit')

                # NEW: Find indifference point (where p(HIT) = 0.5)
                # For logistic regression: p = 1 / (1 + exp(-(intercept + slope*x)))
                # At p=0.5: 0.5 = 1 / (1 + exp(-(intercept + slope*x)))
                # Solving: x = -intercept / slope
                intercept = result.params['const']  # Changed from [0]
                slope = result.params['tokens_before_choice']  # Changed from [1]

                #                 print(f"DEBUG: intercept = {intercept}, slope = {slope}")

                if slope != 0:
                    indifference_point = -intercept / slope
                    #                     print(f"DEBUG: indifference_point = {indifference_point}")

                    # Plot a circle marker at the indifference point
                    ax1.plot(indifference_point, 0.5, 'ro', markersize=10,
                             label=f'Indifference Point: {indifference_point:.2f} tokens',
                             zorder=5)
            except Exception as e:
                print("DEBUG: slope is 0!")
        #                 print(f"DEBUG EXCEPTION: {type(e).__name__}: {e}")  # Better error message
        #                 import traceback
        #                 traceback.print_exc()

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
        ax2.set_title('Reaction Time Distribution: HIT vs STAY', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 3000)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_box_aspect(0.62)

        # ===== SUBPLOT 3: HIT vs STAY Box Plot (Horizontal) =====
        hit_rts = df[df['choice'] == 'HIT']['reaction_time_ms']
        stay_rts = df[df['choice'] == 'STAY']['reaction_time_ms']

        bp = ax3.boxplot([hit_rts, stay_rts], positions=[1, 2], widths=0.5,
                         tick_labels=['HIT', 'STAY'], patch_artist=True, vert=False)

        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')

        ax3.scatter(hit_rts, [1] * len(hit_rts), alpha=0.2, s=15, color='blue')
        ax3.scatter(stay_rts, [2] * len(stay_rts), alpha=0.2, s=15, color='red')

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
            bp4 = ax4.boxplot(rt_data_1, tick_labels=bin_labels_1, patch_artist=True, vert=False)

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
            bp5 = ax5.boxplot(rt_data_5, tick_labels=bin_labels_5_used, patch_artist=True, vert=False)

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
        plt.savefig(os.path.join(self.data_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()


    def plot_adjusted_average_pumps_vs_tokens(self, subject_data):
        """
        Create scatter plot of total tokens collected vs Adjusted Average Pumps for all subjects

        Args:
            subject_data: List of tuples (subject_name, total_tokens, adjusted_avg_pumps)
        """
        subject_names = [s[0][:3] for s in subject_data]
        total_tokens = [s[1] for s in subject_data]
        adjusted_avg_pumps = [s[2] for s in subject_data]

        fig, ax = plt.subplots(figsize=(10, 7))

        # Scatter plot
        scatter = ax.scatter(adjusted_avg_pumps, total_tokens, s=150, alpha=0.6,
                             color='steelblue', edgecolors='navy', linewidth=1.5)

        # Add labels for each point
        for i, name in enumerate(subject_names):
            ax.annotate(name, (adjusted_avg_pumps[i], total_tokens[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

        # Linear regression
        if len(adjusted_avg_pumps) > 2:
            z = np.polyfit(adjusted_avg_pumps, total_tokens, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(adjusted_avg_pumps), max(adjusted_avg_pumps), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label='Linear fit')

            # Calculate correlation
            correlation = np.corrcoef(adjusted_avg_pumps, total_tokens)[0, 1]
            ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Adjusted Average Pumps (Hits in WIN sequences / # WIN sequences)', fontsize=12)
        ax.set_ylabel('Total Tokens Collected', fontsize=12)
        ax.set_title('Total Tokens vs Adjusted Average Pumps (Risk-Taking Index)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'adjusted_average_pumps_vs_tokens.png'), dpi=300, bbox_inches='tight')
        print("Saved: adjusted_average_pumps_vs_tokens.png")
        plt.close()


    def generate_summary_statistics(self, df, output_filename="summary_statistics.txt"):
        """Generate summary statistics with statistical tests"""

        output = []
        output.append("\n" + "=" * 60)
        output.append("SUMMARY STATISTICS")
        output.append("=" * 60)

        # Basic counts
        output.append(f"\nTotal trials: {len(df)}")
        output.append(f"Total sequences: {df['sequence_number'].max() + 1}")
        output.append(f"Total tokens accumulated: {df['total_accumulated'].iloc[-1]}")

        # ===== HIT vs STAY Analysis =====
        output.append("\n" + "-" * 60)
        output.append("HIT vs STAY REACTION TIMES")
        output.append("-" * 60)

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
        cohens_d = (hit_rts.mean() - stay_rts.mean()) / np.sqrt(
            ((len(hit_rts) - 1) * hit_rts.std() ** 2 + (len(stay_rts) - 1) * stay_rts.std() ** 2) / (
                        len(hit_rts) + len(stay_rts) - 2))
        output.append(f"  Cohen's d = {cohens_d:.3f}")

        # Mann-Whitney U test (non-parametric alternative)
        u_stat, p_value_mw = stats.mannwhitneyu(hit_rts, stay_rts)
        output.append(f"\nMann-Whitney U test (non-parametric):")
        output.append(f"  U = {u_stat:.1f}, p = {p_value_mw:.4f}")

        # ===== RT by Tokens at Trial Onset (BINNED) =====
        output.append("\n" + "-" * 60)
        output.append("REACTION TIME BY TOKENS AT TRIAL ONSET (BINNED: 0-5, 6-10, 11-15, 16-20)")
        output.append("-" * 60)

        df_binned = df.copy()
        df_binned['tokens_before_choice'] = df_binned['sequence_total'] - df_binned['tokens_this_hit']

        # Create custom bins: 0-5, 6-10, 11-15, 16-20
        bins = [0, 6, 11, 16, 21]
        bin_labels = ['0-5', '6-10', '11-15', '16-20']
        df_binned['token_bin'] = pd.cut(df_binned['tokens_before_choice'],
                                        bins=bins,
                                        labels=bin_labels,
                                        right=False,
                                        include_lowest=True)
        df_binned = df_binned.dropna(subset=['token_bin'])

        grouped = df_binned.groupby('token_bin', observed=True)['reaction_time_ms'].agg(
            ['mean', 'std', 'count']).reset_index()
        grouped.columns = ['token_bin', 'mean_rt', 'std_rt', 'n_trials']

        output.append("\nToken Bin    | N Trials | Mean RT (ms) | SD (ms)")
        output.append("-" * 50)
        for _, row in grouped.iterrows():
            output.append(
                f"{str(row['token_bin']):12s} | {int(row['n_trials']):8d} | {row['mean_rt']:12.1f} | {row['std_rt']:8.1f}")

        # ===== Kruskal-Wallis test on BINNED data =====
        output.append("\n" + "-" * 60)
        output.append("KRUSKAL-WALLIS TEST: Reaction Time Across Token Bins")
        output.append("-" * 60)

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
        output.append("\n" + "-" * 60)
        output.append("POST-HOC PAIRWISE COMPARISONS (Mann-Whitney U with Bonferroni)")
        output.append("-" * 60)

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
                output.append(
                    f"  • {b1} tokens: RT {direction} than {b2} tokens ({mean_rt_b1:.0f} vs {mean_rt_b2:.0f} ms, p={p:.6f})")
        else:
            output.append(f"\nNo pairwise differences survive Bonferroni correction.")

        output.append("\n" + "=" * 60 + "\n")

        # Save to file
        output_text = "\n".join(output)
        print(output_text)

        with open(os.path.join(self.data_dir, output_filename), 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Saved: {output_filename}")


    def plot_phit_by_deck(self, df, filename="phit_by_deck.png"):
        """Plot P(HIT) vs points accumulated at START of trial, separated by deck type"""

        fig, axes = plt.subplots(4, 1, figsize=(5, 10))
        axes = axes.flatten()

        # Expected values for each deck
        deck_evs = {
            'HIGH': 8.5,
            'MEDIUM': 7.0,
            'LOW': 5.8
        }

        indifference_points = {}

        # Calculate points at START of each trial (sequence_total before this trial's tokens)
        df_copy = df.copy()
        df_copy['points_at_start'] = df_copy['sequence_total'] - df_copy['tokens_this_hit']

        # Define deck order
        deck_types = ['HIGH', 'MEDIUM', 'LOW']

        for idx, deck in enumerate(deck_types):
            df_deck = df_copy[df_copy['deck_type'] == deck]

            # Calculate P(HIT) by points at START of trial
            grouped = df_deck.groupby('points_at_start').agg({
                'choice': lambda x: (x == 'HIT').sum() / len(x)
            }).reset_index()
            grouped.columns = ['points_at_start', 'p_hit']
            grouped['n_trials'] = df_deck.groupby('points_at_start').size().values

            ax = axes[idx]
            ax.scatter(grouped['points_at_start'], grouped['p_hit'],
                       s=grouped['n_trials'] * 30, alpha=0.6)

            # Fit logistic and get indifference point
            fit_result = self.get_binomial_fit(grouped['points_at_start'].values,
                                               grouped['p_hit'].values)

            if fit_result['success']:
                ax.plot(fit_result['x_smooth'], fit_result['y_pred'],
                        "r--", linewidth=2, label='Logistic fit')

                ip = self.get_indifference_point(fit_result['model'])
                if ip is not None:
                    ax.plot(ip, 0.5, 'ro', markersize=10,
                            label=f"IP: {ip:.2f}")
                    indifference_points[deck] = ip
                    ax.legend()

            ax.set_xlabel('Points at Trial Start')
            ax.set_ylabel('P(HIT)')
            ax.set_title(f'{deck} Deck')
            ax.set_ylim(-0.2, 1.2)
            ax.set_xlim(0, 21)
            ax.grid(True, alpha=0.3)

        # Plot 4th subplot: IP vs EV
        ax4 = axes[3]
        evs = [deck_evs[deck] for deck in deck_types if deck in indifference_points]
        ips = [indifference_points[deck] for deck in deck_types if deck in indifference_points]

        ax4.scatter(evs, ips, s=200, alpha=0.6, color='red')
        ax4.plot([0, 15],[0,15], color='black', linestyle='dashed', linewidth=2)
        for deck, ev in deck_evs.items():
            if deck in indifference_points:
                ip = indifference_points[deck]
                ax4.annotate(deck, (ev, ip), xytext=(5, 5),
                             textcoords='offset points', fontsize=10)

        ax4.set_aspect('equal')
        ax4.set_xlabel('Expected Value of Deck')
        ax4.set_ylabel('Indifference Point')
        ax4.set_title('Indifference Point vs Deck EV')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, filename), dpi=150)
        print(f"Saved: {filename}")


    def plot_phit_by_deck_per_subject(self, filename_prefix="phit_by_deck"):
        """Plot P(HIT) vs points accumulated, separated by deck type, for each subject"""

        # Expected values for each deck
        deck_evs = {
            'HIGH': 8.5,
            'MEDIUM': 7.0,
            'LOW': 5.8
        }

        deck_types = ['HIGH', 'MEDIUM', 'LOW']

        # Get unique subjects
        for session_idx, session in enumerate(self.sessions):
            subject_id = session['subject_id']
            df = pd.DataFrame(session['trials'])

            print(f"\nProcessing: {subject_id}")

            fig, axes = plt.subplots(4, 1, figsize=(5, 15))
            axes = axes.flatten()

            indifference_points = {}

            # Calculate points at START of each trial
            df_copy = df.copy()
            df_copy['points_at_start'] = df_copy['sequence_total'] - df_copy['tokens_this_hit']

            for idx, deck in enumerate(deck_types):
                df_deck = df_copy[df_copy['deck_type'] == deck]

                if len(df_deck) == 0:
                    print(f"  No data for {deck} deck")
                    continue

                # Calculate P(HIT) by points at START of trial
                grouped = df_deck.groupby('points_at_start').agg({
                    'choice': lambda x: (x == 'HIT').sum() / len(x)
                }).reset_index()
                grouped.columns = ['points_at_start', 'p_hit']
                grouped['n_trials'] = df_deck.groupby('points_at_start').size().values

                ax = axes[idx]
                ax.scatter(grouped['points_at_start'], grouped['p_hit'],
                           s=grouped['n_trials'] * 30, alpha=0.6)

                # Fit logistic and get indifference point
                fit_result = self.get_binomial_fit(grouped['points_at_start'].values,
                                                   grouped['p_hit'].values)

                if fit_result['success']:
                    ax.plot(fit_result['x_smooth'], fit_result['y_pred'],
                            "r--", linewidth=2, label='Logistic fit')

                    ip = self.get_indifference_point(fit_result['model'])
                    if ip is not None:
                        ax.plot(ip, 0.5, 'ro', markersize=10,
                                label=f"IP: {ip:.2f}")
                        indifference_points[deck] = ip
                        ax.legend()

                ax.set_xlabel('Points at Trial Start')
                ax.set_ylabel('P(HIT)')
                ax.set_title(f'{deck} Deck')
                ax.set_ylim(-0.2, 1.2)
                ax.set_xlim(0, 21)
                ax.grid(True, alpha=0.3)

            # Plot 4th subplot: IP vs EV
            ax4 = axes[3]
            evs = [deck_evs[deck] for deck in deck_types if deck in indifference_points]
            ips = [indifference_points[deck] for deck in deck_types if deck in indifference_points]

            if len(evs) > 0:
                ax4.scatter(evs, ips, s=200, alpha=0.6, color='red')

                for deck, ev in deck_evs.items():
                    if deck in indifference_points:
                        ip = indifference_points[deck]
                        ax4.annotate(deck, (ev, ip), xytext=(5, 5),
                                     textcoords='offset points', fontsize=10)

            ax4.plot([0,15], [0,15], color='black', linestyle='dashed', linewidth=2)
            ax4.set_aspect('equal')
            ax4.set_xlabel('Expected Value of Deck')
            ax4.set_ylabel('Indifference Point')
            ax4.set_title('Indifference Point vs Deck EV')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_filename = f"{filename_prefix}_{subject_id}.png"
            plt.savefig(os.path.join(self.data_dir, plot_filename), dpi=150)
            print(f"Saved: {plot_filename}")
            plt.close()

    def prepare_prospect_theory_data(self, df):
        """
        Organize trial data for prospect theory analysis.

        Args:
            df: DataFrame with trial data

        Returns:
            DataFrame with prospect theory columns added
        """
        df_pt = df.copy()

        # Calculate points before this trial
        df_pt['points_before'] = df_pt['sequence_total'] - df_pt['tokens_this_hit']

        # Calculate gain/loss (tokens earned this trial, or loss if BUST)
        df_pt['gain_loss'] = df_pt.apply(
            lambda row: row['tokens_this_hit'] if row['result'] != 'BUST' else -row['points_before'],
            axis=1
        )

        # Domain: gain (points > 0) or loss (BUST)
        df_pt['domain'] = df_pt.apply(
            lambda row: 'LOSS' if row['result'] == 'BUST' else ('GAIN' if row['points_before'] > 0 else 'NEUTRAL'),
            axis=1
        )

        return df_pt


    def analyze_risk_by_domain(self, df, filename="risk_by_domain.png"):
        """Plot P(HIT) by gain/loss domain"""

        df_pt = self.prepare_prospect_theory_data(df)

        domains = ['LOSS', 'NEUTRAL', 'GAIN']
        p_hits = []

        fig, ax = plt.subplots(figsize=(10, 6))

        for domain in domains:
            df_domain = df_pt[df_pt['domain'] == domain]
            p_hit = (df_domain['choice'] == 'HIT').sum() / len(df_domain)
            p_hits.append(p_hit)
            print(f"{domain}: P(HIT) = {p_hit:.3f}, n = {len(df_domain)}")

        ax.bar(domains, p_hits, color=['red', 'gray', 'green'], alpha=0.6)
        ax.set_ylabel('P(HIT)')
        ax.set_ylim(0, 1)
        ax.set_title('Risk-Taking by Domain (Prospect Theory)')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, filename), dpi=150)
        print(f"Saved: {filename}")

    def analyze_risk_by_domain_per_subject(self, filename_prefix="risk_by_domain"):
        """Analyze risk by domain for each subject"""

        domains = ['LOSS', 'NEUTRAL', 'GAIN']

        for session_idx, session in enumerate(self.sessions):
            subject_id = session['subject_id']
            df = pd.DataFrame(session['trials'])

            print(f"\nProcessing: {subject_id}")

            df_pt = self.prepare_prospect_theory_data(df)

            fig, ax = plt.subplots(figsize=(10, 6))

            p_hits = []
            for domain in domains:
                df_domain = df_pt[df_pt['domain'] == domain]
                if len(df_domain) > 0:
                    p_hit = (df_domain['choice'] == 'HIT').sum() / len(df_domain)
                    p_hits.append(p_hit)
                    print(f"  {domain}: P(HIT) = {p_hit:.3f}, n = {len(df_domain)}")
                else:
                    p_hits.append(0)
                    print(f"  {domain}: No data")

            ax.bar(domains, p_hits, color=['red', 'gray', 'green'], alpha=0.6)
            ax.set_ylabel('P(HIT)')
            ax.set_ylim(0, 1)
            ax.set_title(f'{subject_id} - Risk-Taking by Domain')
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plot_filename = f"{filename_prefix}_{subject_id}.png"
            plt.savefig(os.path.join(self.data_dir, plot_filename), dpi=150)
            print(f"Saved: {plot_filename}")
            plt.close()


    def prospect_value(self, x, alpha, beta, lambda_param):
        """
        Calculate prospect theory value function.

        Args:
            x: gain/loss amount
            alpha: diminishing sensitivity for gains
            beta: diminishing sensitivity for losses
            lambda_param: loss aversion coefficient

        Returns:
            subjective value
        """
        if x >= 0:
            return x ** alpha
        else:
            return -lambda_param * ((-x) ** beta)


    def softmax_choice(self, v_hit, v_stay, gamma):
        """
        Calculate P(HIT) using softmax function.

        Args:
            v_hit: value of hitting
            v_stay: value of staying
            gamma: choice sensitivity (inverse temperature)

        Returns:
            P(HIT) in range [0, 1]
        """
        exp_diff = np.exp(gamma * (v_hit - v_stay))
        return exp_diff / (1 + exp_diff)


    def negative_log_likelihood(self, params, df_pt):
        """
        Calculate negative log-likelihood with improved value model.
        """
        alpha, beta, lambda_param, gamma = params

        # Bounds checking
        if alpha <= 0 or beta <= 0 or lambda_param <= 0 or gamma <= 0:
            return 1e10
        if alpha > 2 or beta > 2 or lambda_param > 5 or gamma > 10:
            return 1e10

        nll = 0
        n_valid = 0

        for idx, row in df_pt.iterrows():
            try:
                tokens_this_trial = row['tokens_this_hit']
                points_before = row['points_before']

                if pd.isna(tokens_this_trial) or tokens_this_trial < 0:
                    continue

                # STAY: gain 0 (secure current points)
                v_stay = self.prospect_value(0, alpha, beta, lambda_param)

                # HIT:
                # - If you win: gain tokens_this_trial
                # - If you bust: lose points_before
                # Assume 90% chance to get tokens, 10% chance to bust (simplified)
                p_win = 0.9
                p_bust = 0.1

                v_hit = p_win * self.prospect_value(tokens_this_trial, alpha, beta, lambda_param) + \
                        p_bust * self.prospect_value(-points_before, alpha, beta, lambda_param)

                # Softmax
                diff = gamma * (v_hit - v_stay)

                if diff > 100:
                    p_hit = 0.9999
                elif diff < -100:
                    p_hit = 0.0001
                else:
                    exp_diff = np.exp(diff)
                    p_hit = exp_diff / (1 + exp_diff)

                p_hit = np.clip(p_hit, 1e-10, 1 - 1e-10)

                if row['choice'] == 'HIT':
                    nll -= np.log(p_hit)
                else:
                    nll -= np.log(1 - p_hit)

                n_valid += 1

            except Exception as e:
                continue

        if n_valid == 0 or np.isnan(nll):
            return 1e10

        return nll


    def fit_prospect_theory_group(self, df, filename="pt_fit_group.txt"):
        """Fit prospect theory parameters to group data"""

        df_pt = self.prepare_prospect_theory_data(df)

        # Initial guess
        x0 = [0.88, 0.88, 2.25, 1.0]

        # Bounds: alpha, beta, lambda, gamma all > 0
        bounds = [(0.1, 2.0), (0.1, 2.0), (0.1, 5.0), (0.1, 10.0)]

        # Optimize
        result = minimize(
            self.negative_log_likelihood,
            x0,
            args=(df_pt,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True}
        )

        alpha, beta, lambda_param, gamma = result.x

        # Print results
        output = [
            "\n" + "=" * 60,
            "PROSPECT THEORY GROUP-LEVEL FIT",
            "=" * 60,
            f"\nOptimization Success: {result.success}",
            f"Final NLL: {result.fun:.4f}",
            f"\nEstimated Parameters:",
            f"  alpha (diminishing sensitivity, gains): {alpha:.4f}",
            f"  beta (diminishing sensitivity, losses): {beta:.4f}",
            f"  lambda (loss aversion): {lambda_param:.4f}",
            f"  gamma (choice sensitivity): {gamma:.4f}",
            f"\nTypical literature values:",
            f"  alpha: ~0.88",
            f"  beta: ~0.88",
            f"  lambda: ~2.25",
            f"  gamma: varies"
        ]

        output_text = "\n".join(output)
        print(output_text)

        with open(os.path.join(self.data_dir, filename), 'w', encoding='utf-8') as f:
            f.write(output_text)


    def fit_prospect_theory_from_indifference_points(self, df, filename="pt_fit_from_ip.txt"):
        """
        Fit prospect theory parameters using indifference points as reference points.
        """

        df_pt = self.prepare_prospect_theory_data(df)

        # Expected values for each deck
        deck_evs = {
            'HIGH': 8.5,
            'MEDIUM': 7.0,
            'LOW': 5.8
        }

        indifference_points = {}
        deck_types = ['HIGH', 'MEDIUM', 'LOW']

        # First, recalculate indifference points from data
        for deck in deck_types:
            df_deck = df_pt[df_pt['deck_type'] == deck]
            grouped = df_deck.groupby('points_at_start').agg({
                'choice': lambda x: (x == 'HIT').sum() / len(x)
            }).reset_index()
            grouped.columns = ['points_at_start', 'p_hit']

            fit_result = self.get_binomial_fit(grouped['points_at_start'].values,
                                               grouped['p_hit'].values)

            if fit_result['success']:
                ip = self.get_indifference_point(fit_result['model'])
                if ip is not None:
                    indifference_points[deck] = ip

        print("Indifference Points (Reference Points):")
        for deck, ip in indifference_points.items():
            print(f"  {deck}: {ip:.2f}")

        # Now fit PT parameters using IP as reference point
        def objective(params):
            alpha, beta, lambda_param, gamma = params

            if alpha <= 0 or beta <= 0 or lambda_param <= 0 or gamma <= 0:
                return 1e10

            error = 0

            for deck in deck_types:
                if deck not in indifference_points:
                    continue

                ref_point = indifference_points[deck]
                df_deck = df_pt[df_pt['deck_type'] == deck]

                for idx, row in df_deck.iterrows():
                    try:
                        tokens_this_trial = row['tokens_this_hit']
                        points_before = row['points_before']

                        # Reframe relative to reference point (IP)
                        gain_hit = tokens_this_trial  # gain from hitting
                        loss_bust = -(points_before)  # loss from busting
                        gain_stay = 0  # no change from staying

                        # Probability of bust (rough estimate: increases with points_before)
                        p_bust = min(0.1 + points_before * 0.02, 0.5)
                        p_win = 1 - p_bust

                        # Value function with reference point
                        v_hit = p_win * self.prospect_value(gain_hit, alpha, beta, lambda_param) + \
                                p_bust * self.prospect_value(loss_bust, alpha, beta, lambda_param)
                        v_stay = self.prospect_value(gain_stay, alpha, beta, lambda_param)

                        # Softmax
                        diff = gamma * (v_hit - v_stay)
                        if diff > 100:
                            p_hit_pred = 0.9999
                        elif diff < -100:
                            p_hit_pred = 0.0001
                        else:
                            exp_diff = np.exp(diff)
                            p_hit_pred = exp_diff / (1 + exp_diff)

                        p_hit_pred = np.clip(p_hit_pred, 1e-10, 1 - 1e-10)

                        # Add to error
                        actual_hit = 1 if row['choice'] == 'HIT' else 0
                        error += (actual_hit - p_hit_pred) ** 2

                    except:
                        continue

            return error

        # Optimize
        x0 = [0.88, 0.88, 2.25, 1.0]
        bounds = [(0.1, 2.0), (0.1, 2.0), (0.1, 5.0), (0.01, 5.0)]

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'disp': True})

        alpha, beta, lambda_param, gamma = result.x

        # Output
        output = [
            "\n" + "=" * 60,
            "PROSPECT THEORY FIT (Using Indifference Points as Reference)",
            "=" * 60,
            f"\nOptimization Success: {result.success}",
            f"Final Error: {result.fun:.4f}",
            f"\nEstimated Parameters:",
            f"  alpha (gains): {alpha:.4f}",
            f"  beta (losses): {beta:.4f}",
            f"  lambda (loss aversion): {lambda_param:.4f}",
            f"  gamma (choice sensitivity): {gamma:.4f}",
            f"\nIndifference Points (Reference Points):",
        ]

        for deck, ip in indifference_points.items():
            output.append(f"  {deck}: {ip:.2f}")

        output.extend([
            f"\nTypical literature values:",
            f"  alpha: ~0.88",
            f"  beta: ~0.88",
            f"  lambda: ~2.25",
            f"  gamma: varies"
        ])

        output_text = "\n".join(output)
        print(output_text)

        with open(os.path.join(self.data_dir, filename), 'w', encoding='utf-8') as f:
            f.write(output_text)


    def fit_prospect_theory_per_deck(self, df, filename="pt_fit_per_deck.txt"):
        """
        Fit prospect theory parameters separately for each deck.
        """

        df_pt = self.prepare_prospect_theory_data(df)

        deck_evs = {
            'HIGH': 8.5,
            'MEDIUM': 7.0,
            'LOW': 5.8
        }

        deck_types = ['HIGH', 'MEDIUM', 'LOW']
        results_dict = {}

        output = [
            "\n" + "=" * 60,
            "PROSPECT THEORY FIT - PER DECK",
            "=" * 60
        ]

        for deck in deck_types:
            df_deck = df_pt[df_pt['deck_type'] == deck]

            # Calculate IP for this deck
            grouped = df_deck.groupby('points_at_start').agg({
                'choice': lambda x: (x == 'HIT').sum() / len(x)
            }).reset_index()
            grouped.columns = ['points_at_start', 'p_hit']

            fit_result = self.get_binomial_fit(grouped['points_at_start'].values,
                                               grouped['p_hit'].values)

            ip = None
            if fit_result['success']:
                ip = self.get_indifference_point(fit_result['model'])

            output.append(f"\n{deck} Deck (EV={deck_evs[deck]}, IP={ip:.2f if ip else 'N/A'})")
            output.append("-" * 40)

            if ip is None:
                output.append("  Could not calculate indifference point")
                continue

            # Fit for this deck
            def objective(params):
                alpha, beta, lambda_param, gamma = params

                if alpha <= 0 or beta <= 0 or lambda_param <= 0 or gamma <= 0:
                    return 1e10

                error = 0
                count = 0

                for idx, row in df_deck.iterrows():
                    try:
                        tokens_this_trial = row['tokens_this_hit']
                        points_before = row['points_before']

                        # Relative to IP
                        gain_hit = tokens_this_trial
                        loss_bust = -points_before

                        p_bust = min(0.05 + points_before * 0.03, 0.6)
                        p_win = 1 - p_bust

                        v_hit = p_win * self.prospect_value(gain_hit, alpha, beta, lambda_param) + \
                                p_bust * self.prospect_value(loss_bust, alpha, beta, lambda_param)
                        v_stay = self.prospect_value(0, alpha, beta, lambda_param)

                        diff = gamma * (v_hit - v_stay)
                        if diff > 100:
                            p_hit_pred = 0.9999
                        elif diff < -100:
                            p_hit_pred = 0.0001
                        else:
                            exp_diff = np.exp(diff)
                            p_hit_pred = exp_diff / (1 + exp_diff)

                        p_hit_pred = np.clip(p_hit_pred, 1e-10, 1 - 1e-10)

                        actual_hit = 1 if row['choice'] == 'HIT' else 0
                        error += (actual_hit - p_hit_pred) ** 2
                        count += 1

                    except:
                        continue

                if count == 0:
                    return 1e10

                return error / count

            x0 = [0.88, 0.88, 2.25, 1.0]
            bounds = [(0.3, 1.5), (0.3, 1.5), (0.5, 4.0), (0.1, 3.0)]

            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

            alpha, beta, lambda_param, gamma = result.x
            results_dict[deck] = {
                'alpha': alpha,
                'beta': beta,
                'lambda': lambda_param,
                'gamma': gamma,
                'error': result.fun,
                'ip': ip
            }

            output.extend([
                f"  alpha: {alpha:.4f}",
                f"  beta: {beta:.4f}",
                f"  lambda: {lambda_param:.4f}",
                f"  gamma: {gamma:.4f}",
                f"  MSE: {result.fun:.6f}"
            ])

        output_text = "\n".join(output)
        print(output_text)

        with open(os.path.join(self.data_dir, filename), 'w', encoding='utf-8') as f:
            f.write(output_text)

        return results_dict


    def run_analysis(self):
        """Run complete analysis: individual plots + summary plots"""
        if len(self.sessions) == 0:
            print("No session data found!")
            return

        print("\nGenerating analysis...\n")

        # Store data for cross-subject comparison
        subject_data = []

        # Generate individual plots for each session
        print("=" * 60)
        print("INDIVIDUAL SESSION ANALYSIS")
        print("=" * 60)

        for idx, session in enumerate(self.sessions):
            subject_name = f"{session['subject_id']}_{session['session_timestamp']}"

            print(f"\nProcessing: {subject_name}")
            df_individual = self.get_session_trials(idx)

            if len(df_individual) == 0:
                print(f"  ⚠ No trials found in this session, skipping...")
                continue

            # Calculate Adjusted Average Pumps
            adjusted_avg_pumps, n_win_sequences = self.calculate_adjusted_average_pumps(df_individual)
            total_tokens = session['total_tokens_accumulated']

            print(f"  Adjusted Average Pumps: {adjusted_avg_pumps:.3f}")
            print(f"  WIN sequences: {n_win_sequences}")
            print(f"  Total tokens collected: {total_tokens}")

            # Store data for cross-subject comparison
            subject_data.append((subject_name, total_tokens, adjusted_avg_pumps))

            # Create individual plot
            plot_filename = f"individual_analysis_{subject_name}.png"
            self.plot_consolidated_analysis(df_individual, filename=plot_filename)

            # Create individual statistics
            stats_filename = f"individual_statistics_{subject_name}.txt"
            self.generate_summary_statistics(df_individual, output_filename=stats_filename)
        # Generate combined summary plots
        print("\n" + "=" * 60)
        print("CONSOLIDATED SUMMARY PLOTS (ALL SESSIONS COMBINED)")
        print("=" * 60 + "\n")

        df_combined = self.combine_all_trials()
        self.plot_consolidated_analysis(df_combined, filename="consolidated_analysis.png")
        self.generate_summary_statistics(df_combined, output_filename="summary_statistics.txt")
        self.plot_phit_by_deck(df_combined, filename="phit_by_deck.png")

        # Generate cross-subject scatter plot
        print("\n" + "=" * 60)
        print("CROSS-SUBJECT ANALYSIS")
        print("=" * 60 + "\n")

        if len(subject_data) > 1:
            self.plot_adjusted_average_pumps_vs_tokens(subject_data)

            # Print cross-subject summary
            print("Subject-level Adjusted Average Pumps Summary:")
            print("-" * 70)
            print(f"{'Subject':<30} | {'Total Tokens':>15} | {'Adj. Avg. Pumps':>15}")
            print("-" * 70)
            for subject_name, total_tokens, adjusted_avg_pumps in subject_data:
                print(f"{subject_name:<30} | {total_tokens:>15} | {adjusted_avg_pumps:>15.3f}")

            self.plot_phit_by_deck_per_subject()

        # Prospect theory analysis
        print("\n" + "=" * 60)
        print("PROSPECT THEORY ANALYSIS")
        print("=" * 60 + "\n")

        df_combined = self.combine_all_trials()

        # 1. Domain analysis (already working great)
        self.analyze_risk_by_domain(df_combined)
        self.analyze_risk_by_domain_per_subject()

        # 2. Parameter fitting (just per-deck, which is most informative)
        df_combined = self.prepare_prospect_theory_data(df_combined)
        self.fit_prospect_theory_per_deck(df_combined)


if __name__ == "__main__":
    analyzer = pBARTAnalyzer()
    analyzer.run_analysis()

