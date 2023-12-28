import numpy

class Matrice :
    matrice : list[list[float]]


    def simpleMatrice(lin : int, col : int):
        retMat = Matrice (lin, col)
        minimum = min(lin, col)
        for i in range(minimum):
            retMat.matrice[i][i] = 1
        return retMat
    
    def matrice(mat : list[list[float]]):
        retMat = Matrice(len(mat), len(mat[0]))
        retMat.matrice = mat
        return retMat
    
    def randomMatrice(lin : int, col : int):
        retMat = Matrice(lin, col)
        retMat.matrice = numpy.random.uniform(-1 ,1, size=(lin, col))
        return retMat

    def __init__(self, lin : int, col : int) -> None:
        self.matrice = []
        for l in range(lin) :
            self.matrice.append([])
            for c in range(col) : 
                self.matrice[l].append(0)

    def setMat(self, mat : list[list[float]]) -> None:
        self.matrice = mat

    def transpose(mat : 'Matrice'):
        transposedMat = Matrice(mat.columns(), mat.lines())
        for l in range(mat.lines()):
            for c in range(mat.columns()):
                transposedMat.matrice[c][l] = mat.matrice[l][c]
        return transposedMat

        
    def multiply(mat1 : 'Matrice', mat2 : 'Matrice'):
        if(not mat1.columns() == mat2.lines()) :
            print ("Le nombre de colonnes" , mat1.columns(),  "n'est pas egale au nombre de lignes", mat2.lines())
        else :
            retMat = Matrice(mat1.lines(), mat2.columns())
            for l in range(retMat.lines()):
                for c in range(retMat.columns()):
                    for i in range (mat1.columns()):
                        retMat.matrice[l][c] += mat1.matrice[l][i] * mat2.matrice[i][c] 
            return retMat
        
    def activate(self):
        for l in range(self.lines()) :
            for c in range(self.columns()) : 
                self.matrice[l][c] = numpy.sign(self.matrice[l][c])
    
    def map(mat : 'Matrice', function):
        retMat = Matrice(mat.lines(), mat.columns())
        for l in range(mat.lines()) :
            for c in range(mat.columns()) : 
                retMat.matrice[l][c] = function(mat.matrice[l][c])
        return retMat
        

    def add(mat1 : 'Matrice', mat2 : 'Matrice'):
        retMat = Matrice(mat1.lines(), mat1.columns())
        for l in range(retMat.lines()):
            for c in range(retMat.columns()):
                retMat.matrice[l][c] =mat1.matrice[l][c] + mat2.matrice[l][c]
        return retMat

    def substract(mat1 : 'Matrice', mat2 : 'Matrice'):
        retMat = Matrice(mat1.lines(), mat1.columns())
        for l in range(retMat.lines()):
            for c in range(retMat.columns()):
                retMat.matrice[l][c] =mat1.matrice[l][c] - mat2.matrice[l][c]
        return retMat
        
    def multiplyNb(mat : 'Matrice', nb: int):
        retMat = Matrice(mat.lines(), mat.columns())
        for l in range(retMat.lines()):
            for c in range(retMat.columns()):
                retMat.matrice[l][c] =mat.matrice[l][c] * nb
        return retMat
    

    def multiplySimple(mat1 : 'Matrice', mat2 : 'Matrice'):
        retMat = Matrice(mat1.lines(), mat1.columns())
        for l in range(retMat.lines()):
            for c in range(retMat.columns()):
                retMat.matrice[l][c] =mat1.matrice[l][c] * mat2.matrice[l][c]
        return retMat



    def lines(self) :
        return len(self.matrice)
    
    def columns(self) :
        return len(self.matrice[0])
    

    def print(self):
        print(self.matrice)

