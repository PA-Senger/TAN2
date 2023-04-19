import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.integrate as spi

# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

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

A_ = np.dot(P, D)
A_ = np.dot(A_, Pinv)

print("P = \n", P)
print("D = \n", D)
print("P^{-1} = \n", Pinv)
# on remarque que les int de A ont été cast en float c'est pas grave
print("\nPAP^{-1} = \n", A_)
# e_j est un vecteur propre associé à la valeur propre de la j-eme colonne de D


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
print("\n Puissance itérée :")
print("Plus grande valeur propre : %.4f" % lam1)
print("Vecteur propre associé : ", v1)

# Plus petite valeur propre et vecteur propre associé
lam2, v2 = inverse_power_iteration(A)
print("Plus petite valeur propre : %.4f" % lam2)
print("Vecteur propre associé : ", v2)


### Question 2 ###

def gamma(t):
    return np.exp((2j*np.pi*t))  # j est l'unité imaginaire en python


t = np.linspace(0, 1, 1000)     # subdivsion de [0;1]
z = gamma(t)                    # notre nombre complexe evalué sur la sub [0,1]
x, y = np.real(z), np.imag(z)
plt.title("Courbe paramétrée")
plt.plot(x, y)
# plt.show()


### Question 4 ###

def dgamma(t):
    """ Dérivé de la fonction gamma définie plus haut."""
    return 1 / (2j*np.pi) * gamma(t)


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

print("\n Intégrales curvilignes complexe : ")
print("I_1: ", I_1, " ,Erreur d'integration: ", err1)
print("I_2: ", I_2, " ,Erreur d'integration: ", err2)


### Question 5 ###

def rectangle_gauche(f, a, b, N):
    """ Intégration numérique par la méthode des rectangles à gauches.

    Parameters:
    -----------
    f : fonction à valeurs complexe.
    a,b : floats.
        Bornes d'intégrations.
    N : int. 
        Nombre de pas dans la subdivisions.

    Returns
    ------
    res : complexe.
        La valeur approcher de l'intégrale de f sur [a,b].
    """
    res = 0.0
    h = (b - a) / float(N)  # pas de la subdivision

    for i in range(N):
        x = a + i*h     # i commence à 0, on prend le point à gauche du rectangle
        res += f(x) * h  # hauteur * base

    return res


J_1 = rectangle_gauche(f1, 0, 1, 10000)
J_2 = rectangle_gauche(f2, 0, 1, 10000)

print("\n Méthodes des rectangles à gauches : ")
print("J_1 = ", J_1)
print("J_2 = ", J_2)

# La méthode est d'ordre >1 car d'apres la question 3, gamma est une fonction
# régulière définie sur l'intervalle [0,1] et d'apres le papier cité en source
# du sujet (https://irma.math.unistra.fr/~helluy/PREPRINTS/cras1998.pdf)
# on peut utiliser la méthode dite de "périodisation"


### Question 6 ###

t = sp.Symbol('t', real=True)
gamma = 2 * sp.exp(2 * sp.I * sp.pi * t)
# dgamma = sp.diff(gamma, t)
dgamma = 4 * sp.I * sp.pi * sp.exp(2 * sp.I * sp.pi * t)

z = sp.Symbol('z')
A_symb = sp.Matrix([[3, 3, 4], [6, -2, -12], [-2, 3, 9]])
I_symb = sp.Matrix.eye(3)
f = sp.Matrix(z * I_symb - A_symb).inv()
h = f.subs(z, gamma)
int_f = 1 / (2*sp.I*sp.pi) * sp.integrate(h * dgamma, t)

print("Intégrale symbolique : \n")
sp.pprint(sp.simplify(int_f))
