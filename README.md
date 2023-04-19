# TAN2


Person: 
- Écrire un programme Python utilisant numpy qui permet de retrouver la
plus grande et la plus petite valeur propre d’une matrice A au moyen de
la méthode la puissance itérée, ainsi que des vecteurs propres associés

ChatGPT: 
- Voici un exemple de programme Python qui utilise numpy pour calculer la plus grande et la plus petite valeur propre d'une matrice A en utilisant la méthode de la puissance itérée :

```python

import numpy as np

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

    n = A.shape[0]
    v = np.random.rand(n)
    lam = 0.0

    for i in range(max_iter):
        v_new = np.linalg.solve(A, v)
        lam_new = np.linalg.norm(v_new)
        v_new = v_new / lam_new

        if np.abs(lam_new - lam) < tol:
            break

        v = v_new
        lam = lam_new

    return (1/lam, v)

# Exemple d'utilisation
A = np.array([[4, 2], [2, 3]])

# Plus grande valeur propre et vecteur propre associé
lam1, v1 = power_iteration(A)
print("Plus grande valeur propre :", lam1)
print("Vecteur propre associé :", v1)

# Plus petite valeur propre et vecteur propre associé
lam2, v2 = inverse_power_iteration(A)
print("Plus petite valeur propre :", lam2)
print("Vecteur propre associé :", v2)
```



Dans cet exemple, la matrice d'entrée `A` est définie comme une matrice `2x


https://fr.wikipedia.org/wiki/M%C3%A9thodes_de_calcul_d%27int%C3%A9grales_de_contour



- Person:
corrige cette fonction : 
def integrale_curviligne(f, gamma, dgamma, a=0, b=1):
    """ 
    L'intégrale curviligne de f le long de gamma.

    Parameters: 
    -----------
    f : fonction.
    gamma : fonction.
        Chemin de classe C1 par morceaux.
    Dgamma : fonction.
        Dérivée de gamma.
    a,b : float.
        Borne d'integrations, domaine de gamma.

    Returns:
    -------
    (res, err) : couple of (complexe, float).
        Valeur de l'intégrale de f le long de gamma et l'erreur d'integration.
    """
    Re = np.real(f(gamma(t)) * dgamma(t))
    Im = np.imag(f(gamma(t)) * dgamma(t))

    res1, e1 = sc.integrate.quad(lambda t: Re, a, b)
    res2, e2 = sc.integrate.quad(lambda t: Im, a, b)
    return res1 + 1j*res2, e1+e2


- ChatGpt:
Il y a plusieurs erreurs dans cette fonction. Voici une version corrigée :

```python
import numpy as np
import scipy.integrate as spi

def integrale_curviligne(f, gamma, dgamma, a=0, b=1):
    """ 
    L'intégrale curviligne de f le long de gamma.

    Parameters: 
    -----------
    f : fonction.
    gamma : fonction.
        Chemin de classe C1 par morceaux.
    dgamma : fonction.
        Dérivée de gamma.
    a,b : float.
        Borne d'intégrations, domaine de gamma.

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
```

Voici les corrections que j'ai apportées :

- J'ai ajouté l'importation de la bibliothèque `scipy.integrate` sous le nom `spi`, pour éviter de devoir taper `scipy.integrate` à chaque fois.
- J'ai corrigé l'orthographe de `dgamma` dans la liste des paramètres.
- J'ai défini une fonction `integrande` qui calcule l'intégrande `f(gamma(t)) * dgamma(t)`.
- J'ai utilisé la fonction `quad` de la bibliothèque `scipy.integrate` pour calculer l'intégrale curviligne de la partie réelle et de la partie imaginaire de `integrande`.
- J'ai calculé l'intégrale curviligne complexe en combinant les résultats de l'intégration de la partie réelle et de la partie imaginaire, et j'ai calculé l'erreur en combinant les erreurs de l'intégration de la partie réelle et de la partie imaginaire.