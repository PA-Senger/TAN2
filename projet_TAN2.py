import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.integrate as spi

# np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

### Question 1 ###

A = np.array([[3, 3, 4], [6, -2, -12], [-2, 3, 9]])

vp, vep = np.linalg.eig(A)

# A est scindé à racines simples, donc diagonalisable dans R
print("Valeurs propres de A : ", vp)

# on trie les valeurs propres en gardant l'ordre des vecteurs propres associés
idx = vp.argsort()[::1]
vp = vp[idx]
vep = vep[:, idx]

D = np.diag(vp, 0)
P = np.array(vep)
Pinv = np.linalg.inv(P)

A_ = P @ D @ Pinv  # @ : dot product

print("\nP = \n", np.around(P, 2))
print("\nD = \n", np.around(D, 2))
print("\nP^{-1} = \n", np.around(Pinv, 2))
# on remarque que les int de A ont été cast en float c'est pas grave
print("\nPAP^{-1} = \n", A_)
# e_j est un vecteur propre associé à la valeur propre de la j-eme colonne de D


# on met tout dans une fonction pour l'utilisé dans la derniere question
def diagonalize(A):
    """"Diagonalise A en mettant les valeurs propre dans l'ordre croissant.
    Attention la fonction ne vérifie pas que A est diagonalisable !

    Parameters:
    ----------
    A : array.
        Matrice à diagonalisé.

    Returns:
    -------
    (D, P) : couple of arrays.
        D est la matrice diagonale avec les vp triée ,ie d1,1 < d2,2 < .. < dn,n.
        P est la matrice de passage.
    """
    vp, vep = np.linalg.eig(A)
    idx = vp.argsort()[::1]
    vp = vp[idx]
    vep = vep[:, idx]
    D = np.diag(vp, 0)
    P = np.array(vep)
    return D, P


### Question 2 ###

def power_iteration(A, max_iter=1000, tol=1e-8):
    """
    Calcule la plus grande valeur propre de la matrice A
    ainsi que le vecteur propre associé en utilisant la méthode
    de la puissance itérée.

    Parameters:
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

    n = A.shape[0]        # extrait la taille de A pour créer le vecteur propre
    v = np.random.rand(n)  # on part d'un vecteur propre random
    lam = 0.0             # initialisation de lambda la vp recherché

    for i in range(max_iter):             # algorithme de la puissance itérée
        v_new = A @ v                     # v_k+1 = A dot v_k
        lam_new = np.linalg.norm(v_new)
        v_new = v_new / lam_new           # v_k+1 = v_k+1 / norm(v_k+1)

        if np.abs(lam_new - lam) < tol:   # critere d'arret de précision
            break

        v = v_new       # mise à jour des valeurs
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

    n = A.shape[0]        # extrait la taille de A pour créer le vecteur propre
    v = np.random.rand(n)  # on par d'un vecteur propre random
    lam = 0.0

    for i in range(max_iter):
        v_new = np.linalg.solve(A, v)
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
print("\n--------------------\n")
print("Puissance itérée :")
print("Plus grande valeur propre : ", np.around(lam1, 4))
print("Vecteur propre associé : ", np.around(v1, 4))

# Plus petite valeur propre et vecteur propre associé
lam2, v2 = inverse_power_iteration(A)
print("\nPlus petite valeur propre : ", np.around(lam2, 4))
print("Vecteur propre associé : ", np.around(v2, 4))


### Question 3 ###

def gamma(t):
    return np.exp((2j*np.pi*t))  # j est l'unité imaginaire en python


t = np.linspace(0, 1, 1000)     # subdivsion de [0;1]
z = gamma(t)                    # notre nombre complexe evalué sur la sub [0,1]
x, y = np.real(z), np.imag(z)
plt.title("Courbe paramétrée de Gamma")
plt.xlabel("Re")
plt.ylabel("Img")
plt.plot(x, y)
# plt.show()


### Question 4 ###

def dgamma(t):
    """ Dérivé de la fonction gamma définie plus haut."""
    return 2j*np.pi * gamma(t)


def integrale_curviligne(f, gamma, dgamma, a=0, b=1):
    """
    L'intégrale curviligne de f le long de gamma.

    Parameters:
    -----------
    f : fonction complexe.
    gamma : fonction.
        Chemin de classe C1 par morceaux.
    dgamma : fonction.
        Dérivée de gamma.
    a,b : float.
        Bornes d'intégrations, domaine de gamma.

    Returns:
    -------
    (res, err) : couple of (complexe, float).
        Valeur de l'intégrale de f le long de gamma et l'erreur d'integration.
    """
    def integrande(t):
        return f(gamma(t)) * dgamma(t)

    # Intégration de la partie réelle de l'intégrale curviligne
    Re, Re_err = spi.quad(lambda t: np.real(integrande(t)), a, b)

    # Intégration de la partie imaginaire de l'intégrale curviligne
    Im, Im_err = spi.quad(lambda t: np.imag(integrande(t)), a, b)

    # Calcul de l'intégrale curviligne complexe et de l'erreur
    res = Re + 1j * Im
    err = np.sqrt(Re_err**2 + Im_err**2)

    return res, err


def f1(z):
    z0 = 0.5 * (1 + 1j)
    return (z - z0)**3


def f2(z):
    z0 = 0.5 * (1 + 1j)
    return 1. / (z - z0)


I_1, err1 = integrale_curviligne(f1, gamma, dgamma, 0, 1)
I_2, err2 = integrale_curviligne(f2, gamma, dgamma, 0, 1)
print("\n--------------------\n")
print("Intégrales curvilignes complexe : ")
print("Jc(f1): ", I_1, " ,Erreur d'integration: ", err1)
print("Jc(f2): ", I_2, " ,Erreur d'integration: ", err2)


### Question 5 ###

def rectangle_gauche(f, a, b, N=1000):
    """ Intégration numérique par la méthode des rectangles à gauches.

    Parameters:
    -----------
    f : fonction à valeurs complexe.
    a,b : floats.
        Bornes d'intégrations.
    N : int.
        Nombre de pas dans la subdivision.

    Returns
    ------
    res : complexe.
        La valeur approcher de l'intégrale de f sur [a,b].
    """
    res = 0j
    h = (b - a) / float(N)  # pas de la subdivision

    for i in range(N):
        x = a + i*h     # i commence à 0, on prend le point à gauche du rectangle
        res += f(x) * h  # hauteur * base

    return res


# Test
def exp(x):
    return np.exp(x)


print("\n--------------------\n")
# e-1 ~= 1.718
print("Test rectangle gauche : ", rectangle_gauche(exp, 0, 1, 10000))

J_1 = rectangle_gauche(lambda t: f1(gamma(t))*dgamma(t), 0, 1, 10000)
J_2 = rectangle_gauche(lambda t: f2(gamma(t))*dgamma(t), 0, 1, 10000)
print("\nMéthodes des rectangles à gauches : ")
print("J(f1) = ", J_1)
print("J(f2)= ", J_2)

# La méthode est d'ordre >1 car d'apres la question 3, gamma est une fonction
# régulière définie sur l'intervalle [0,1] et d'apres le papier cité en source
# du sujet (https://irma.math.unistra.fr/~helluy/PREPRINTS/cras1998.pdf)
# on peut utiliser la méthode dite de "périodisation"


### Question 6 ###

# Pour la partie gauche, on calcul l'intégrale curviligne

t = sp.Symbol('t')
# Chemin gamma
gamma = 2 * sp.exp(2 * sp.I * sp.pi * t)
# La dérivé de gamma
dgamma = 4 * sp.I * sp.pi * sp.exp(2 * sp.I * sp.pi * t)

# Notre matrice A
A_symb = sp.Matrix([[3, 3, 4], [6, -2, -12], [-2, 3, 9]])

B = gamma * sp.eye(3) - A_symb
f = B.inv() * dgamma

# L'intégrale curviligne recherché
int_f = (1 / (2*sp.I*sp.pi)) * sp.integrate(f, (t, 0, 1))

print("\n--------------------\n")
print("Intégrale symbolique : \n")
sp.pprint(int_f)

# Pour la partie droite, on doit créer la projection

# La matrice J_1 de l'énoncer
J = sp.Matrix.zeros(3)
J[0, 0] = 1

# Pour obtenir P et P**-1 on diagonalise A avec sympy

# Diagonalisation de la matrice A
# Les vp sont dans l'ordre croissant
P, D = A_symb.diagonalize(sort=True)
# sp.pprint(D)

# Vérification :
# sp.pprint(P * D * P**-1)

Pi_1 = P * J * P**-1
sp.pprint(Pi_1)


### Question 7 ###

# on va crée notre matrice intégrande puis utiliser la methode des rectangles
# à gauche pour intégrer chaque entrée de la matrice

def contour(gamma, dgamma, A, a, b, N=1000):
    """Calcule la formule de l'integrale de contour d'une matrice.

    Parameters:
    -----------
    gamma : function.
        Chemin C1 par morceaux
    dgamma : function.
        Dérivée de gamma.
    A : array.
        Matrice.
    a,b : float. 
        Borne d'intégration.
    N : int. 
        Nombre de pas dans la subdivision.

    Returns:
    --------
        res : array.
            Matrice des intégrales.
    """
    n, m = np.shape(A)
    res = np.zeros((n, m), dtype=complex)

    def integrande(t):
        return np.linalg.inv(gamma(t)*np.eye(n) - A) * dgamma(t)

    for i in range(n):
        for j in range(m):  # c'est une matrice carré de toute facon mais bon
            res[i, j] = rectangle_gauche(
                lambda t: integrande(t)[i, j], a, b, N)

    return res


# Chemin gamma
def gamma2(t):
    return 2 * np.exp(2j * np.pi * t)


# La dérivée de gamma
def dgamma2(t):
    return 4j * np.pi * np.exp(2j * np.pi * t)


contour = contour(gamma2, dgamma2, A, 0, 1, 1000)
print("\n--------------------\n")
print("Contour numérique : \n")
print(contour)


def projection(A, K):
    """Calcule la formule de la somme sur k dans K des projections PI_k.

    Parameters:
    ----------
    A : array.
        Matrice.
    K : array.
        Veteur contenant les indices.

    Returns:
    -------
        res : array.
            Matrice de la somme des projections.
    """
    def proj(i, A):
        D, P = diagonalize(A)
        n, m = np.shape(P)
        J = np.zeros((n, m))
        J[i-1, i-1] = 1
        proj = P @ J @ np.linalg.inv(P)
        return proj
    res = np.zeros(np.shape(A))
    for i in range(len(K)):
        res += proj(K[i], A)
    return res


print("\nProjection : \n")
K = [1]
print(projection(A, K))
