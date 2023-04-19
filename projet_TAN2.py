import numpy as np

### Question 1 ###

A = np.array([[3, 3, 4], [6, -2, -12], [-2, 3, 9]])

vp, vep = np.linalg.eig(A)

# A est scindé à racines simples, donc diagonalisable dans R
print("Valeurs propre: ", vp)

# on trie les valeurs propre en gardant l'ordre des vecteurs propres associés
idx = vp.argsort()[::1]
vp = vp[idx]
vep = vep[:, idx]

D = np.diag(vp, 0)
P = np.array(vep)
P_ = np.linalg.inv(P)

A_ = np.dot(P, D)
A_ = np.dot(A_, P_)

print("D = \n", D)
# on remarque que les int de A ont été cast en float
print("\nPAP^ = \n", A_)
# e_j est un vecteur propre associé à la valeur propre de la j-eme colonne de D


### Question 2 ###

def power_iteration(A, max_iter=1000, tol=1e-8):
    """
    Calcule la plus grande valeur propre de la matrice A
    ainsi que le vecteur propre associé en utilisant la méthode
    de la puissance itérée.

    Parameters
    ----------
    A : numpy.ndarray
        La matrice d'entrée de taille (n, n)
    max_iter : int, optional
        Le nombre maximum d'itérations de la méthode de la puissance itérée.
        Default is 1000.
    tol : float, optional
        La tolérance pour la convergence de la méthode de la puissance itérée.
        Default is 1e-8.

    Returns
    -------
    (lam, v) : tuple of (float, numpy.ndarray)
        La plus grande valeur propre de A et le vecteur propre associé.
    """

    n = A.shape[0]
    v = np.random.rand(n)
    lam = 0.0

    for i in range(max_iter):
        v_new = A @ v
        lam_new = np.linalg.norm(v_new)
        v_new = v_new / lam_new

        if np.abs(lam_new - lam) < tol:
            break

        v = v_new
        lam = lam_new

    return (lam, v)


def inverse_power_iteration(A, max_iter=1000, tol=1e-8):
    """
    Calcule la plus petite valeur propre de la matrice A
    ainsi que le vecteur propre associé en utilisant la méthode
    de la puissance itérée inverse.

    Parameters
    ----------
    A : numpy.ndarray
        La matrice d'entrée de taille (n, n)
    max_iter : int, optional
        Le nombre maximum d'itérations de la méthode de la puissance itérée inverse.
        Default is 1000.
    tol : float, optional
        La tolérance pour la convergence de la méthode de la puissance itérée inverse.
        Default is 1e-8.

    Returns
    -------
    (lam, v) : tuple of (float, numpy.ndarray)
        La plus petite valeur propre de A et le vecteur propre associé.
    """

    # extrait la taille de A pour créer le vecteur propre
    n = A.shape[0]
    v = np.random.rand(n)  # on par d'un vecteur propre random
    lam = 0.0

    for i in range(max_iter):
        v_new = np.linalg.solve(A, v)        # v_k+1 = A * v_k
        lam_new = np.linalg.norm(v_new)
        v_new = v_new / lam_new              # v_k+1 / norme(v_k+1)

        if np.abs(lam_new - lam) < tol:      # critere d'arret de précision
            break

        v = v_new
        lam = lam_new

    return (1/lam, v)


# On test avec A de la question 1
# Plus grande valeur propre et vecteur propre associé
lam1, v1 = power_iteration(A)
print("Plus grande valeur propre :", lam1)
print("Vecteur propre associé :", v1)

# Plus petite valeur propre et vecteur propre associé
lam2, v2 = inverse_power_iteration(A)
print("Plus petite valeur propre :", lam2)
print("Vecteur propre associé :", v2)
