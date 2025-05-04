import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================
# Advanced Data Preprocessing
# ==========================
def advanced_data_preprocessing(df):
    """
    Robust data preprocessing for market basket analysis with auto-scaling
    """
    # Identify potential transaction columns
    def is_transaction_column(series):
        return (series.nunique() > 1 and 
                series.dtype == 'object' and 
                not series.str.contains(',').all())
    
    # Select transaction columns
    transaction_columns = df.apply(is_transaction_column)
    transaction_df = df.loc[:, transaction_columns]
    
    # If no suitable columns found, use all object columns
    if transaction_df.empty:
        transaction_df = df.select_dtypes(include=['object'])
    
    # Auto-scaling parameters
    total_items = transaction_df.shape[1]
    total_transactions = transaction_df.shape[0]
    
    # Dynamically adjust max_items and max_transactions
    max_items = min(total_items, 500)  # Cap at 500 or dataset size
    max_transactions = min(total_transactions, 5000)  # Cap at 5000 or dataset size
    
    # Limit number of columns and rows
    if transaction_df.shape[1] > max_items:
        transaction_df = transaction_df.iloc[:, :max_items]
    
    if transaction_df.shape[0] > max_transactions:
        transaction_df = transaction_df.sample(n=max_transactions, random_state=42)
    
    return transaction_df

def extract_transactions(df):
    """
    Robust transaction extraction
    """
    def process_row(row):
        return [str(val).strip() for val in row if pd.notna(val) and str(val).strip()]
    
    transactions = df.apply(process_row, axis=1).tolist()
    transactions = [trans for trans in transactions if trans]
    
    return transactions

# ==========================
# Advanced Market Basket Analysis
# ==========================
def perform_market_basket_analysis(transactions):
    """
    Comprehensive market basket analysis with auto-tuned parameters
    """
    # Auto-tune min_support based on dataset size
    dataset_size = len(transactions)
    min_support = max(0.001, min(0.1, 10 / dataset_size))
    
    mlb = MultiLabelBinarizer()
    
    try:
        encoded_array = mlb.fit_transform(transactions)
        df_encoded = pd.DataFrame(
            encoded_array, 
            columns=mlb.classes_
        )
        
        results = {}
        rules = {}
        
        # Apriori
        try:
            apriori_results = apriori(df_encoded, min_support=min_support, use_colnames=True)
            results['Apriori'] = apriori_results
            
            if not apriori_results.empty:
                apriori_rules = association_rules(apriori_results, metric="confidence", min_threshold=0.5)
                rules['Apriori'] = apriori_rules
        except Exception as e:
            st.warning(f"Apriori Algorithm failed: {e}")
        
        # FP-Growth
        try:
            fpgrowth_results = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
            results['FP-Growth'] = fpgrowth_results
            
            if not fpgrowth_results.empty:
                fpgrowth_rules = association_rules(fpgrowth_results, metric="confidence", min_threshold=0.5)
                rules['FP-Growth'] = fpgrowth_rules
        except Exception as e:
            st.warning(f"FP-Growth Algorithm failed: {e}")
        
        return results, rules, mlb.classes_
    
    except Exception as e:
        st.error(f"Market Basket Analysis failed: {e}")
        return None, None, None

# ==========================
# Visualization Functions
# ==========================
def visualize_market_basket_comparison(results):
    """
    Create comprehensive comparison visualization
    """
    plt.figure(figsize=(15, 8))
    
    # Prepare data for plotting
    plot_data = []
    for algo, result in results.items():
        if not result.empty:
            result_copy = result.copy()
            result_copy['Algorithm'] = algo
            plot_data.append(result_copy)
    
    if not plot_data:
        st.warning("No data available for visualization")
        return
    
    combined_df = pd.concat(plot_data)
    
    # Multi-panel visualization
    plt.subplot(2, 2, 1)
    # Top frequent itemsets comparison
    top_itemsets = combined_df.sort_values('support', ascending=False).head(10)
    sns.barplot(
        x='support', 
        y='itemsets', 
        hue='Algorithm', 
        data=top_itemsets
    )
    plt.title('Top 10 Frequent Itemsets')
    plt.xlabel('Support')
    plt.ylabel('Itemsets')
    
    plt.subplot(2, 2, 2)
    # Distribution of support across algorithms
    sns.boxplot(x='Algorithm', y='support', data=combined_df)
    plt.title('Support Distribution')
    plt.ylabel('Support')
    
    plt.subplot(2, 2, 3)
    # Itemset count comparison
    itemset_counts = combined_df.groupby('Algorithm').size()
    itemset_counts.plot(kind='bar')
    plt.title('Number of Itemsets per Algorithm')
    plt.ylabel('Count')
    plt.xlabel('Algorithm')
    
    plt.subplot(2, 2, 4)
    # Unique itemsets comparison
    unique_itemsets = combined_df.groupby('Algorithm')['itemsets'].nunique()
    unique_itemsets.plot(kind='bar')
    plt.title('Unique Itemsets per Algorithm')
    plt.ylabel('Unique Itemsets')
    plt.xlabel('Algorithm')
    
    plt.tight_layout()
    st.pyplot(plt)

# ==========================
# Main Streamlit Application
# ==========================
def main():
    st.set_page_config(layout="wide")
    st.title("🛒 Smart Market Basket Analysis")
    
    # File Upload
    uploaded_file = st.file_uploader(
        "Upload Market Basket Dataset", 
        type=['csv', 'xlsx'], 
        help="Upload a dataset for market basket analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Read File
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            # Preprocess Data
            processed_df = advanced_data_preprocessing(df)
            
            # Extract Transactions
            transactions = extract_transactions(processed_df)
            
            if transactions:
                # Perform Market Basket Analysis
                results, rules, unique_items = perform_market_basket_analysis(transactions)
                
                if results:
                    # Create Tabs for Results
                    tab1, tab2, tab3 = st.tabs([
                        "Frequent Itemsets", 
                        "Association Rules", 
                        "Visualization"
                    ])
                    
                    with tab1:
                        st.subheader("Frequent Itemsets")
                        for algo, result in results.items():
                            st.write(f"### {algo} Results")
                            st.dataframe(result)
                    
                    with tab2:
                        st.subheader("Association Rules")
                        for algo, rule in rules.items():
                            st.write(f"### {algo} Rules")
                            st.dataframe(rule)
                    
                    with tab3:
                        st.subheader("Market Basket Algorithm Comparison")
                        visualize_market_basket_comparison(results)
                    
                    # Dataset Insights
                    st.subheader("Dataset Insights")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Unique Items", len(unique_items))
                    with col2:
                        st.metric("Total Transactions", len(transactions))
                    
                    st.success("Market Basket Analysis Completed Successfully!")
                else:
                    st.error("Failed to perform market basket analysis.")
            else:
                st.error("No valid transactions found in the dataset.")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":

    main()
