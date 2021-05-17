import numpy
from math import sqrt


class Color:
   GREY = '\033[38;5;243m'
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def transpose(a, at, x):
    for ii in range(x):
        for j in range(x):
            at[ii][j] = a[j][ii]


def is_symmetric(m, x):
    for ii in range(x):
        for j in range(x):
            if m[ii][j] != m[j][ii]:
                return False
    return True


def is_pos_def(m, x):
    if not is_symmetric(m, x):
        return False
    else:
        # Check if all eigenvalues are positive.
        return numpy.all(numpy.linalg.eigvals(m) > 0)


def cholesky_decomposition(m, x):
    A = numpy.copy(m)
    L = [[0.0] * n for ii in range(x)]

    # Perform the Cholesky decomposition
    for i in range(x):
        for k in range(i + 1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))

            if i == k:  # Diagonal elements
                L[i][k] = sqrt(A[i][i] - tmp_sum)
            else:
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))

    print(Color.BOLD + Color.YELLOW + " L:" + Color.END)
    for i in range(x):
        print(end='\t')
        for j in range(x):
            print("{:.4f}".format(L[i][j]), end='  ')
        print()
    print(Color.BOLD + Color.YELLOW + " U:" + Color.END)
    for i in range(x):
        print(end='\t')
        for j in range(x):
            print("{:.4f}".format(L[j][i]), end='  ')
        print()
    return L


def lu_decomposition(m, x):
    L = [[0 for i in range(x)]for j in range(x)]
    U = [[0 for i in range(x)]for j in range(x)]

    for i in range(x):
        for k in range(i, x):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
            U[i][k] = m[i][k] - sum

        for k in range(i, x):
            if i == k:
                L[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])
                L[k][i] = int((m[k][i] - sum) / U[i][i])

    print(Color.BOLD + Color.YELLOW + " L:" + Color.END)
    for i in range(x):
        print(end='\t')
        for j in range(x):
            print("{:.4f}".format(L[i][j]), end='  ')
        print()
    print(Color.BOLD + Color.YELLOW + " U:" + Color.END)
    for i in range(x):
        print(end='\t')
        for j in range(x):
            print("{:.4f}".format(U[i][j]), end='  ')
        print()
    return L, U


if __name__ == "__main__":
    n = int(input(Color.BOLD + "Enter size of your Square matrix: " + Color.END))
    M = []
    L = []
    U = []
    print(Color.BOLD + "Enter your matrix:" + Color.END)
    print(Color.GREY + "Example for size 2:\n2 1\n"
          "1 3" + Color.END)
    for i in range(n):
       M.append(list(map(int, input().split())))

    if is_pos_def(M, n):
        print(Color.BLUE + "Your Matrix is Positive Definite" + Color.END)
        print(Color.BOLD + Color.UNDERLINE + Color.RED +
              "Cholesky Decomposition:" + Color.END)
        L = cholesky_decomposition(M, n)
        U = [[0] * n for ii in range(n)]
        transpose(L, U, n)

    else:
        print(Color.BLUE + "Your Matrix is" + Color.END +
                Color.RED + " NOT " + Color.END +
                Color.BLUE + "Positive Definite" + Color.END)
        print(Color.BOLD + Color.UNDERLINE + Color.RED +
              "LU Decomposition:" + Color.END)
        L, U = lu_decomposition(M, n)

    # Hale Dastgah!
    b = list(map(int, input(Color.BOLD + "\nEnter b to solve Ax=b: " + Color.END).split()))
    Y = numpy.linalg.solve(L, b)
    X = numpy.linalg.solve(U, Y)
    print(Color.BOLD + Color.YELLOW + "X =" + Color.END + " (", end='')
    for i in X:
        print("{:.4f},".format(i), end=' ')
    print("\b\b)")
