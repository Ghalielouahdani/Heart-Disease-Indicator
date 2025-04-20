import argparse
import numpy as np
from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.knn import KNN
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, mse_fn
import os
import matplotlib.pyplot as plt

np.random.seed(100)

def find_best_lr_logistic(xtrain, ytrain, xval, yval):
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    best_lr = None
    best_acc = -np.inf
    results = {}

    print("\nRecherche du meilleur learning rate pour LogisticRegression :\n")
    for lr in learning_rates:
        model = LogisticRegression(lr=lr)
        model.fit(xtrain, ytrain)
        preds = model.predict(xval)
        acc = accuracy_fn(preds, yval)
        results[lr] = acc
        print(f"  lr = {lr:.0e} --> accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_lr = lr

    print(f"\nMeilleur learning rate = {best_lr:.0e} avec accuracy = {best_acc:.4f}")
    return best_lr

def try_logistic_regression_grid(xtrain, ytrain, xval, yval):
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    max_iters_list = [50, 100, 200, 500, 1000]
    acc_grid = np.zeros((len(learning_rates), len(max_iters_list)))
    f1_grid = np.zeros((len(learning_rates), len(max_iters_list)))

    print("\nGrid search for lr and max_iters:\n")
    for i, lr in enumerate(learning_rates):
        for j, max_iters in enumerate(max_iters_list):
            print(f"Training LogisticRegression with lr={lr}, max_iters={max_iters}")
            model = LogisticRegression(lr=lr, max_iters=max_iters, reg_strength=0.1)
            model.fit(xtrain, ytrain)
            preds = model.predict(xval)
            acc = accuracy_fn(preds, yval)
            f1 = macrof1_fn(preds, yval)
            acc_grid[i, j] = acc
            f1_grid[i, j] = f1
            print(f"  Validation Accuracy: {acc:.4f}, F1: {f1:.6f}")

    # Plotting heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, grid, title in zip(axes, [acc_grid, f1_grid], ["Validation Accuracy", "Validation F1-score"]):
        im = ax.imshow(grid, aspect='auto', cmap='coolwarm')
        ax.set_xticks(np.arange(len(max_iters_list)))
        ax.set_yticks(np.arange(len(learning_rates)))
        ax.set_xticklabels(max_iters_list, rotation=45, ha='right')
        ax.set_yticklabels(learning_rates)
        ax.set_xlabel("max_iters")
        ax.set_ylabel("learning rate")
        ax.set_title(title)
        # Highlight best
        best = np.unravel_index(np.argmax(grid), grid.shape)
        ax.add_patch(plt.Rectangle((best[1]-0.5, best[0]-0.5), 1, 1,
                                  edgecolor='red', facecolor='none', linewidth=2))
        # Annotate values
        for ii in range(grid.shape[0]):
            for jj in range(grid.shape[1]):
                val = grid[ii, jj]
                txt = f"{val:.2f}" if title == "Accuracy" else f"{val:.4f}"
                ax.text(jj, ii, txt, ha='center', va='center', color='white', fontsize=5)
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    # Return best hyperparameters based on accuracy
    best_idx = np.unravel_index(np.argmax(acc_grid), acc_grid.shape)
    return learning_rates[best_idx[0]], max_iters_list[best_idx[1]]

def main(args):
    """
    Fonction principale améliorée qui inclut :
    - La création d'un ensemble de validation (si args.test n'est pas activé)
    - Une recherche d'hyperparamètre pour la valeur de k via --grid_search pour knn et kmeans
    - L'évaluation sur l'ensemble de validation (ou test si spécifié)
    """
    ## 1. Chargement des données
    if args.data_type == "features":
        feature_data = np.load("Data_MS1_2025/features.npz", allow_pickle=True)
        xtrain, xtest = feature_data["xtrain"], feature_data["xtest"]
        ytrain, ytest = feature_data["ytrain"], feature_data["ytest"]
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path, "dog-small-64")
        xtrain, xtest, ytrain, ytest = load_data(data_dir)

    ## 2. Préparation des données (création d'un ensemble de validation, sélection de caractéristiques et normalisation)
    if not args.test:
        indices = np.arange(xtrain.shape[0])
        np.random.shuffle(indices)
        xtrain, ytrain = xtrain[indices], ytrain[indices]
        
        def create_validation_set(X, y, val_size=0.2):
            n_val = int(len(X) * val_size)
            indices = np.random.permutation(len(X))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            return X[train_indices], X[val_indices], y[train_indices], y[val_indices]
        xtrain, xval, ytrain, yval = create_validation_set(xtrain, ytrain)

        # Calculate statistics from training data
        means = np.mean(xtrain, axis=0, keepdims=True)
        stds = np.std(xtrain, axis=0, keepdims=True)
        
        # Normalize all data using training statistics
        xtrain = normalize_fn(xtrain, means, stds)
        xval = normalize_fn(xval, means, stds)
        xtest = normalize_fn(xtest, means, stds)
        
        
    ## 3. Recherche d'hyperparamètres pour k (pour knn et kmeans)
    if args.grid_search and args.method in ["knn", "kmeans"]:
        candidate_ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        best_k = None
        best_f1 = -np.inf
        best_acc = -np.inf
        results = {}
        for k_val in candidate_ks:
            if args.method == "knn":
                model = KNN(k=k_val)
            elif args.method == "kmeans":
                model = KMeans(n_clusters=k_val, max_iter=args.max_iters)
            
            preds_train = model.fit(xtrain, ytrain)
            # Évaluation sur l'ensemble de validation
            preds_val = model.predict(xval)
            acc = accuracy_fn(preds_val, yval)
            f1 = macrof1_fn(preds_val, yval)
            results[k_val] = {"accuracy": acc, "f1": f1}
            print(f"Pour k = {k_val}: Validation set: accuracy = {acc:.3f}% - F1-score = {f1:.6f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_acc = acc
                best_k = k_val
        
        print(f"\nMeilleur k sur validation: {best_k} avec accuracy = {best_acc:.3f}% et F1 = {best_f1:.6f}")
        # Affichage du meilleur k en fonction de l'accuracy seule
        best_acc_k = max(results.items(), key=lambda x: x[1]["accuracy"])
        print(f"Meilleur k selon l'accuracy seule : {best_acc_k[0]} avec accuracy = {best_acc_k[1]['accuracy']:.3f}%")
        # Utiliser le meilleur k pour le modèle final
        if args.method == "knn":
            method_obj = KNN(k=best_k)
        elif args.method == "kmeans":
            method_obj = KMeans(n_clusters=best_k, max_iter=args.max_iters)
    elif args.grid_search and args.method == "logistic_regression":
        best_lr, best_max_iters = try_logistic_regression_grid(xtrain, ytrain, xval, yval)
        method_obj = LogisticRegression(lr=best_lr, max_iters=best_max_iters, reg_strength=0.1)
    else:
        # Choix du modèle selon l'argument direct
        if args.method == "dummy_classifier":
            method_obj = DummyClassifier(arg1=1, arg2=2)
        elif args.method == "knn":
            method_obj = KNN(k=args.K)
        elif args.method == "kmeans":
            method_obj = KMeans(n_clusters=args.K, max_iter=args.max_iters)
        elif args.method == "logistic_regression":
            method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters, reg_strength=0.1)
        else:
            raise ValueError(f"Unknown method: {args.method}")

    ## 4. Entraînement et évaluation
    preds_train = method_obj.fit(xtrain, ytrain)
    preds_train = method_obj.predict(xtrain)
    if args.test:
        preds = method_obj.predict(xtest)
        target_set = "Test set"
        true_labels = ytest
    else:
        preds = method_obj.predict(xval)
        target_set = "Validation set"
        true_labels = yval

    train_acc = accuracy_fn(preds_train, ytrain)
    train_f1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {train_acc:.3f}% - F1-score = {train_f1:.6f}")
    
    set_acc = accuracy_fn(preds, true_labels)
    set_f1 = macrof1_fn(preds, true_labels)
    print(f"{target_set}: accuracy = {set_acc:.3f}% - F1-score = {set_f1:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="dummy_classifier", type=str, 
                        help="dummy_classifier / knn / logistic_regression / kmeans / nn (MS2)")
    parser.add_argument("--data_path", default="data", type=str, help="path to your dataset")
    parser.add_argument("--data_type", default="features", type=str, help="features/original(MS2)")
    parser.add_argument("--K", type=int, default=1, 
                        help="Nombre de voisins pour knn ou nombre de clusters pour kmeans")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate pour les méthodes utilisant un LR")
    parser.add_argument("--max_iters", type=int, default=100, help="Nombre max d'itérations pour les méthodes itératives")
    parser.add_argument("--test", action="store_true", 
                        help="Entraîner sur l'ensemble complet de formation et évaluer sur le test, sinon utiliser un ensemble de validation")
    parser.add_argument("--grid_search", action="store_true", 
                        help="Effectuer une recherche d'hyperparamètre pour k (pour knn et kmeans)")
    
    # Arguments pour MS2
    parser.add_argument("--nn_type", default="cnn", help="Réseau à utiliser, peut être 'Transformer' ou 'cnn'")
    parser.add_argument("--nn_batch_size", type=int, default=64, help="Batch size pour l'entraînement du réseau de neurones")
    args = parser.parse_args()
    main(args)