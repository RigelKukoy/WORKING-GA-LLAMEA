
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from iohblade.behaviour_metrics import compute_behavior_metrics

def test_projection():
    print("Creating dummy code data...")
    # Dummy code snippets
    codes = [
        "import numpy as np\nx = 1",
        "import numpy as np\nx = 2",
        "import pandas as pd\ny = 3",
        "def foo():\n return 1",
        "class Model:\n pass",
        "x = np.array([1,2,3])",
        "y = np.array([4,5,6])"
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'evaluations': range(len(codes)),
        'raw_y': np.random.rand(len(codes)),
        'code': codes
    })
    
    print("Projecting code to coordinates...")
    # 1. TF-IDF
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(codes)
    
    # 2. SVD (LSA) to reduce to 2 dimensions (for testing)
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)
    
    # 3. Add to DataFrame
    df['x0'] = X_reduced[:, 0]
    df['x1'] = X_reduced[:, 1]
    
    print("Coordinates added. First few rows:")
    print(df[['x0', 'x1']].head())
    
    print("\nComputing metrics...")
    metrics = compute_behavior_metrics(df)
    print("Metrics computed successfully!")
    print(metrics)

if __name__ == "__main__":
    test_projection()
