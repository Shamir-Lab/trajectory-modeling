"""
Shared utility functions for MIMIC-III trajectory analysis notebooks.

This module provides common functions for:
- Data loading and preprocessing
- Feature engineering
- Visualization
- Statistical testing
- Model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def aggregate_vitals_labs(vitals_df, labs_df, vital_meta, labs_meta):
    """
    Aggregate vitals and labs to daily min/max/mean.
    
    Parameters
    ----------
    vitals_df : pd.DataFrame
        Raw vitals data with columns: hadm_id, charttime, itemid, valuenum, admittime
    labs_df : pd.DataFrame
        Raw labs data with same structure
    vital_meta : pd.DataFrame
        Vital signs metadata
    labs_meta : pd.DataFrame
        Labs metadata
        
    Returns
    -------
    pd.DataFrame
        Combined aggregated vitals and labs with hadm_id, admittime, charttime (daily), 
        and min/max/mean columns for all features
    """
    # Merge with metadata and validate
    vitals_df = vitals_df.merge(vital_meta, on='itemid', how='left')
    vitals_df = vitals_df[vitals_df['valuenum'].between(vitals_df['min'], vitals_df['max'], inclusive='both')]
    
    labs_df = labs_df.merge(labs_meta, on='itemid', how='left')
    labs_df = labs_df[labs_df['valuenum'].between(labs_df['min'], labs_df['max'], inclusive='both')]
    
    # Convert fahrenheit to celsius for temperature
    if 'units' in vitals_df.columns:
        vitals_df.loc[vitals_df['units'] == 'F', 'valuenum'] = (
            vitals_df.loc[vitals_df['units'] == 'F', 'valuenum'] - 32
        ) * 5.0/9.0
        vitals_df.loc[vitals_df['units'] == 'F', 'units'] = 'C'
        vitals_df.loc[vitals_df['feature name'] == 'TempF', 'feature name'] = 'TempC'
    
    # Concatenate vitals and labs BEFORE pivoting
    vitals_labs = pd.concat([vitals_df, labs_df], ignore_index=True)
    
    # Pivot with daily aggregation (min, max, mean)
    vitals_labs_pivot = vitals_labs.pivot_table(
        index=['hadm_id', 'admittime', pd.Grouper(freq='1D', key='charttime')],
        columns='feature name',
        values='valuenum',
        aggfunc=['min', 'max', 'mean']
    )
    
    # Flatten column names: feature_stat format
    vitals_labs_pivot.columns = [f'{col[1]}_{col[0]}' for col in vitals_labs_pivot.columns]
    
    return vitals_labs_pivot


def standardize_feature_names(df, feature_map):
    """
    Standardize feature names across datasets.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features to rename
    feature_map : dict
        Mapping from original names to standardized names
        
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names
    """
    return df.rename(columns=feature_map)


def drop_high_missing_columns(df, threshold=0.7, exclude_cols=None):
    """
    Drop columns with missing values above threshold after forward fill.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    threshold : float
        Maximum allowed missing rate (0-1)
    exclude_cols : list, optional
        Columns to exclude from dropping
        
    Returns
    -------
    pd.DataFrame
        DataFrame with high-missing columns removed
    dropped_cols : list
        List of dropped column names
    """
    if exclude_cols is None:
        exclude_cols = ['hadm_id', 'time_day', 'target']
    
    missing_rates = df.drop(columns=exclude_cols, errors='ignore').isna().mean()
    high_missing = missing_rates[missing_rates > threshold].index.tolist()
    
    df_clean = df.drop(columns=high_missing)
    
    return df_clean, high_missing


def impute_features(df, hadm_col='hadm_id', traj_cols=None, vital_cols=None, lab_cols=None):
    """
    Impute missing values with intelligent forward filling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (must have hadm_col as index or column)
    hadm_col : str
        Name of patient/admission ID column
    traj_cols : list, optional
        Trajectory probability columns (ffill limit=2)
    vital_cols : list, optional
        Vital sign columns (ffill limit=3)
    lab_cols : list, optional
        Lab value columns (ffill limit=3)
        
    Returns
    -------
    pd.DataFrame
        Imputed dataframe
    """
    df = df.copy()
    
    # Ensure index is set for groupby
    if hadm_col in df.columns:
        df = df.set_index(hadm_col)
    
    # Trajectory features: ffill limit=2, then 0
    if traj_cols is not None:
        for col in traj_cols:
            if col in df.columns:
                df[col] = df.groupby(level=0)[col].ffill(limit=2)
                df[col] = df[col].fillna(0)

    # Vitals: ffill limit=3
    if vital_cols is not None:
        for col in vital_cols:
            if col in df.columns:
                df[col] = df.groupby(level=0)[col].ffill(limit=3)

    # Labs: ffill limit=3
    if lab_cols is not None:
        for col in lab_cols:
            if col in df.columns:
                df[col] = df.groupby(level=0)[col].ffill(limit=3)
    
    return df


def biomarker_summary_stats(ts_df, value_col, lookback_days):
    """
    Generate summary statistics for biomarker time series over a lookback window.
    
    Works with both MIMIC (hadm_id) and HiRiD (patientid) datasets.
    
    Parameters
    ----------
    ts_df : pd.DataFrame
        Time series data with columns: [hadm_id or patientid, time_day, time_days, value_col]
    value_col : str
        Name of the biomarker column
    lookback_days : int
        Number of days to look back for aggregation
        
    Returns
    -------
    pd.DataFrame
        Summary statistics per patient/admission and timepoint with columns:
        - {id_col}: patient/admission ID
        - time_day: integer day
        - {value_col}_mean_{lookback_days}d
        - {value_col}_max_{lookback_days}d
        - {value_col}_min_{lookback_days}d
        - {value_col}_change_{lookback_days}d
        - {value_col}_linear_trend_{lookback_days}d
        - {value_col}_std_{lookback_days}d
    """
    # Detect patient ID column
    id_col = 'hadm_id' if 'hadm_id' in ts_df.columns else 'patientid'
    
    summary_df_list = []
    for group, group_df in ts_df.groupby(id_col):
        group_df = group_df.sort_values('time_days')
        summary_list = []
        for current_time in group_df['time_day'].unique():
            window_start = current_time - lookback_days
            window_data = group_df[
                group_df['time_day'].between(window_start, current_time, inclusive='both')
            ]
            if len(window_data) > 0:
                value_mean = window_data[value_col].mean()
                value_max = window_data[value_col].max()
                value_min = window_data[value_col].min()
                value_change = window_data[value_col].iloc[-1] - window_data[value_col].iloc[0]
                value_std = window_data[value_col].std()
                
                # Compute linear trend robustly
                try:
                    if len(window_data) >= 2 and value_std > 0:
                        # Use polyfit with error handling
                        value_linear_trend = np.polyfit(window_data['time_days'], window_data[value_col], 1)[0]
                    else:
                        value_linear_trend = np.nan
                except (np.linalg.LinAlgError, ValueError):
                    # If polyfit fails, use simple slope calculation
                    try:
                        x = window_data['time_days'].values
                        y = window_data[value_col].values
                        if len(x) >= 2 and len(np.unique(x)) >= 2:
                            value_linear_trend = (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else np.nan
                        else:
                            value_linear_trend = np.nan
                    except:
                        value_linear_trend = np.nan
            else:
                value_mean = np.nan
                value_max = np.nan
                value_min = np.nan
                value_change = np.nan
                value_linear_trend = np.nan
                value_std = np.nan
            summary_list.append({
                id_col: group,
                'time_day': current_time,
                f'{value_col}_mean_{lookback_days}d': value_mean,
                f'{value_col}_max_{lookback_days}d': value_max,
                f'{value_col}_min_{lookback_days}d': value_min,
                f'{value_col}_change_{lookback_days}d': value_change,
                f'{value_col}_linear_trend_{lookback_days}d': value_linear_trend,
                f'{value_col}_std_{lookback_days}d': value_std
            })
        summary_df_list.append(pd.DataFrame(summary_list))
    return pd.concat(summary_df_list, ignore_index=True)


# ============================================================================
# OUTCOME DEFINITION
# ============================================================================

def keep_first_positive_event(outcome_df, hadm_col='hadm_id', target_col='target', time_col='time_day'):
    """
    Keep all negative windows but only first positive event per patient.
    
    Parameters
    ----------
    outcome_df : pd.DataFrame
        Outcome dataframe with prediction windows
    hadm_col : str
        Patient/admission ID column
    target_col : str
        Target outcome column
    time_col : str
        Time column for sorting
        
    Returns
    -------
    pd.DataFrame
        Filtered outcome dataframe
    """
    outcome_df = outcome_df.sort_values([hadm_col, time_col])
    positive_first = outcome_df[outcome_df[target_col] == 1].groupby(hadm_col).first().reset_index()
    all_negatives = outcome_df[outcome_df[target_col] == 0]
    outcome_df = pd.concat([all_negatives, positive_first], ignore_index=True).sort_values([hadm_col, time_col]).reset_index(drop=True)
    
    return outcome_df


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_trajectory_distribution(df, traj_col='dominant_traj', biomarker_col=None, biomarker_name='Biomarker'):
    """
    Visualize distribution of trajectory types.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with trajectory classifications
    traj_col : str
        Column with trajectory type labels
    biomarker_col : str, optional
        Biomarker column to show mean values
    biomarker_name : str
        Display name for biomarker
        
    Returns
    -------
    fig, ax
        Matplotlib figure and axis
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    traj_counts = df[traj_col].value_counts()
    colors = plt.cm.Set2.colors[:len(traj_counts)]
    
    ax1.bar(range(len(traj_counts)), traj_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(traj_counts)))
    ax1.set_xticklabels([t.replace('_', ' ').title() for t in traj_counts.index], rotation=45, ha='right')
    ax1.set_ylabel('Number of Observations', fontsize=12)
    ax1.set_title('Trajectory Type Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add percentages
    for i, (traj, count) in enumerate(traj_counts.items()):
        pct = 100 * count / len(df)
        ax1.text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Biomarker distribution by trajectory
    if biomarker_col is not None and biomarker_col in df.columns:
        df_clean = df.dropna(subset=[biomarker_col])
        trajectories = df_clean[traj_col].unique()
        
        bp = ax2.boxplot(
            [df_clean[df_clean[traj_col] == t][biomarker_col].values for t in trajectories],
            labels=[t.replace('_', ' ').title() for t in trajectories],
            patch_artist=True,
            widths=0.6,
            showmeans=True
        )
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xticklabels([t.replace('_', ' ').title() for t in trajectories], rotation=45, ha='right')
        ax2.set_ylabel(f'{biomarker_name} Level', fontsize=12)
        ax2.set_title(f'{biomarker_name} by Trajectory Type', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def get_significance_marker(p_value):
    """Return significance marker based on p-value."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


def add_pairwise_brackets(ax, data, pairs_to_compare, y_max):
    """Add bracket lines connecting compared pairs with significance markers."""
    feature_names = list(data.keys()) if isinstance(data, dict) else range(len(data))
    
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    bracket_height_step = y_range * 0.08
    
    for idx, (i, j) in enumerate(pairs_to_compare):
        # Perform Wilcoxon signed-rank test (paired test for CV folds)
        stat, p_value = wilcoxon(data[i], data[j])
        sig_marker = get_significance_marker(p_value)
        
        if sig_marker:  # Only draw bracket if significant
            bracket_y = y_max + bracket_height_step * (idx + 1)
            
            # Draw horizontal line connecting the two boxes
            ax.plot([i + 1, j + 1], [bracket_y, bracket_y], 'k-', linewidth=1.5)
            # Draw vertical ticks at ends
            ax.plot([i + 1, i + 1], [bracket_y - 0.01, bracket_y], 'k-', linewidth=1.5)
            ax.plot([j + 1, j + 1], [bracket_y - 0.01, bracket_y], 'k-', linewidth=1.5)
            # Add significance marker
            ax.text((i + j) / 2 + 1, bracket_y + 0.01, sig_marker, 
                   ha='center', va='bottom', fontsize=12, fontweight='bold')


def plot_boxplots_with_stats(comparison_results, outcome_df, target_col, pairs_to_compare, n_folds_total,
                             roc_ylim=[0, 1.05], ap_ylim=[0, 1.05]):
    """
    Plot boxplots with statistical significance markers.
    
    Parameters
    ----------
    comparison_results : dict
        Dictionary of model results
    outcome_df : pd.DataFrame
        Outcome dataframe for baseline
    target_col : str
        Target column name
    pairs_to_compare : list of tuples
        Pairs of model indices to compare
    n_folds_total : int
        Total number of CV folds
        
    Returns
    -------
    fig, (ax1, ax2)
        Figure and axes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    feature_names = list(comparison_results.keys())
    
    roc_data = [metrics['roc_auc'] for metrics in comparison_results.values()]
    ap_data = [metrics['avg_precision'] for metrics in comparison_results.values()]
    
    # ROC-AUC boxplot
    bp1 = ax1.boxplot(roc_data, labels=feature_names, patch_artist=True,
                       widths=0.6, showmeans=True, meanline=True, showfliers=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7, edgecolor='black', linewidth=1.5),
                       medianprops=dict(color='red', linewidth=2),
                       meanprops=dict(color='blue', linewidth=2, linestyle='--'),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
    
    ax1.set_xticklabels(feature_names, rotation=45, ha='right')
    ax1.set_ylabel('AUROC', fontsize=12)
    ax1.set_title(f'AUROC Distribution Across Feature Sets (n={n_folds_total} folds)', fontsize=14, fontweight='bold')
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random', alpha=0.7)
    
    y_max_roc = max([max(d) for d in roc_data])
    add_pairwise_brackets(ax1, roc_data, pairs_to_compare, y_max_roc)
    ax1.set_ylim(roc_ylim)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Average Precision boxplot
    bp2 = ax2.boxplot(ap_data, labels=feature_names, patch_artist=True,
                       widths=0.6, showmeans=True, meanline=True, showfliers=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7, edgecolor='black', linewidth=1.5),
                       medianprops=dict(color='red', linewidth=2),
                       meanprops=dict(color='blue', linewidth=2, linestyle='--'),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
    
    ax2.set_xticklabels(feature_names, rotation=45, ha='right')
    ax2.set_ylabel('AUPR', fontsize=12)
    ax2.set_title(f'AUPR Distribution Across Feature Sets (n={n_folds_total} folds)', fontsize=14, fontweight='bold')
    baseline = outcome_df[target_col].mean()
    ax2.axhline(y=baseline, color='gray', linestyle='--', linewidth=1, label=f'Baseline ({baseline:.3f})', alpha=0.7)
    
    y_max_ap = max([max(d) for d in ap_data])
    add_pairwise_brackets(ax2, ap_data, pairs_to_compare, y_max_ap)
    ax2.set_ylim(ap_ylim)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def print_statistical_comparisons(comparison_results, pairs_to_compare):
    """
    Print detailed statistical comparison table.
    
    Parameters
    ----------
    comparison_results : dict
        Dictionary of model results
    pairs_to_compare : list of tuples
        Pairs of model indices to compare
    """
    feature_names = list(comparison_results.keys())
    roc_data = [metrics['roc_auc'] for metrics in comparison_results.values()]
    ap_data = [metrics['avg_precision'] for metrics in comparison_results.values()]
    
    print("\n" + "="*80)
    print("PAIRWISE STATISTICAL COMPARISONS (Wilcoxon Signed-Rank Test)")
    print("="*80)
    print(f"{'Comparison':<60} {'AUROC p':<12} {'Sig':<5} {'AUPR p':<12} {'Sig':<5}")
    print("-"*80)
    
    for i, j in pairs_to_compare:
        model_i = feature_names[i]
        model_j = feature_names[j]
        
        # ROC-AUC comparison
        stat_roc, p_roc = wilcoxon(roc_data[i], roc_data[j])
        sig_roc = get_significance_marker(p_roc)
        delta_roc = np.mean(roc_data[j]) - np.mean(roc_data[i])
        
        # Average Precision comparison
        stat_ap, p_ap = wilcoxon(ap_data[i], ap_data[j])
        sig_ap = get_significance_marker(p_ap)
        delta_ap = np.mean(ap_data[j]) - np.mean(ap_data[i])
        
        comparison_str = f"{model_i} vs {model_j}"
        print(f"{comparison_str:<60} {p_roc:<12.4f} {sig_roc:<5} {p_ap:<12.4f} {sig_ap:<5}")
        print(f"  {'Effect size (Δ):':<58} {delta_roc:+.4f}        {delta_ap:+.4f}")
    
    print("="*80)
    print("Note: * p<0.05, ** p<0.01, *** p<0.001")
    print("Wilcoxon test is appropriate here as models are evaluated on identical CV folds (paired data)")
    print("="*80)


def plot_roc_pr_curves(comparison_results, outcome_df, target_col, color_scheme='tab10'):
    """
    Plot ROC and PR curves with SD shading and individual fold curves.
    
    Parameters
    ----------
    comparison_results : dict
        Dictionary of model results
    outcome_df : pd.DataFrame
        Outcome dataframe for baseline
    target_col : str
        Target column name
    color_scheme : str
        Matplotlib colormap name
        
    Returns
    -------
    fig, (ax1, ax2)
        Figure and axes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = getattr(plt.cm, color_scheme).colors
    
    for (feature_set_name, metrics), color in zip(comparison_results.items(), colors):
        # Plot individual fold ROC curves in light color
        for fpr, tpr in metrics['fold_roc_curves']:
            ax1.plot(fpr, tpr, color=color, alpha=0.15, linewidth=1)
        
        # Plot mean ROC curve with SD shading
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        
        for fpr, tpr in metrics['fold_roc_curves']:
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(metrics['roc_auc'])
        std_auc = np.std(metrics['roc_auc'])
        
        ax1.plot(mean_fpr, mean_tpr, color=color, 
                 label=f'{feature_set_name} (AUROC = {mean_auc:.3f} ± {std_auc:.3f})',
                 linewidth=2.5)
        ax1.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, 
                          color=color, alpha=0.2)
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Plot Precision-Recall curves with SD shading and fold curves
    for (feature_set_name, metrics), color in zip(comparison_results.items(), colors):
        # Plot individual fold PR curves in light color
        for recall, precision in metrics['fold_pr_curves']:
            ax2.plot(recall, precision, color=color, alpha=0.15, linewidth=1)
        
        # Plot mean PR curve with SD shading
        mean_recall = np.linspace(0, 1, 100)
        precisions = []
        
        for recall, precision in metrics['fold_pr_curves']:
            precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
        
        mean_precision = np.mean(precisions, axis=0)
        std_precision = np.std(precisions, axis=0)
        mean_ap = np.mean(metrics['avg_precision'])
        std_ap = np.std(metrics['avg_precision'])
        
        ax2.plot(mean_recall, mean_precision, color=color,
                 label=f'{feature_set_name} (AUPR = {mean_ap:.3f} ± {std_ap:.3f})',
                 linewidth=2.5)
        ax2.fill_between(mean_recall, mean_precision - std_precision, mean_precision + std_precision,
                          color=color, alpha=0.2)
    
    baseline = outcome_df[target_col].mean()
    ax2.plot([0, 1], [baseline, baseline], 'k--', linewidth=1, label=f'Baseline ({baseline:.3f})')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig, (ax1, ax2)


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_repeated_cv(prediction_df, feature_cols, target_col, group_col, n_repeats=10, n_folds=5, random_state=920):
    """
    Train model with repeated cross-validation.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    groups : pd.Series
        Group labels for GroupKFold
    n_repeats : int
        Number of CV repetitions
    n_folds : int
        Number of folds per repetition
    random_state : int
        Random seed
        
    Returns
    -------
    dict
        Dictionary with fold metrics
    """
    fold_metrics = {
        'roc_auc': [], 'avg_precision': [],
        'y_true': [], 'y_proba': [],
        'fold_roc_curves': [], 'fold_pr_curves': []
    }
    
    for repeat in range(n_repeats):
        gkf = GroupKFold(n_splits=n_folds)
        
        # Shuffle groups for each repeat
        shuffled_df = prediction_df.sample(frac=1, random_state=920 + repeat * 37)
        X_shuffled = shuffled_df[feature_cols]
        y_shuffled = shuffled_df[target_col]
        groups_shuffled = shuffled_df[group_col]
        
          
        for train_idx, test_idx in gkf.split(X_shuffled, y_shuffled, groups_shuffled):
            scaler = StandardScaler()
            X_train, X_test = X_shuffled.iloc[train_idx], X_shuffled.iloc[test_idx]
            
            # Median impute by X_train medians
            for col in X_train.columns:
                impute_val = X_train[col].median()
                X_train[col] = X_train[col].fillna(impute_val)
                X_test[col] = X_test[col].fillna(impute_val)

            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
            
            y_train, y_test = y_shuffled.iloc[train_idx], y_shuffled.iloc[test_idx]
            
            model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
            random_state=920 + repeat
            )

            model.fit(X_train, y_train)
            
            y_proba = model.predict_proba(X_test)[:, 1]
            
            if y_test.sum() > 0:
                fold_metrics['roc_auc'].append(roc_auc_score(y_test, y_proba))
                fold_metrics['avg_precision'].append(average_precision_score(y_test, y_proba))
                
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fold_metrics['fold_roc_curves'].append((fpr, tpr))
                
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                fold_metrics['fold_pr_curves'].append((recall, precision))
            
            fold_metrics['y_true'].extend(y_test.values)
            fold_metrics['y_proba'].extend(y_proba)
    
    return fold_metrics
