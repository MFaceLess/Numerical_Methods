#include <QCoreApplication>
#include <iostream>
#include <vector>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <QTextStream>
#include <QFile>
#include <QDebug>

#define n_rows 4
#define n_cols 3

using namespace std;

const QString allFileToString(QFile &aFile)
{
    if (!aFile.open(QFile::ReadOnly | QFile::Text)) {
        std::cout << "Error opening file!" << std::endl;
        return NULL;
    }
    QTextStream in(&aFile);
    return in.readAll();
}

void setDataToVector(const QStringList &aStringList,
                     vector< vector <double> > &aVector)
{
    for (size_t row = 0;row < (aStringList.size() - 1);row++)
    {
        QStringList stroka = aStringList.at(row).split(" ");
        vector<double> simple;
        for (size_t i =0;i<stroka.size();i++)
        {
            if ((stroka[i] != "") && (stroka[i] != "\t"))
            {
                simple.push_back(stroka[i].toDouble());
            }
        }
        aVector.push_back(simple);
    }
}

void printVector(const QVector< QVector <double> > &aVector)
{
    for(auto i=aVector.begin();i != aVector.end();i++)
    {
        for(auto j=i->begin();j != i->end();j++)
        {
            cout <<scientific << setw(20) << *j;
        }
        cout << endl;
    }
    cout << endl;
}

void sum(vector<vector<double>> &matrix1,vector<vector<double>> matrix2);
void printMatrix(vector<vector<double>> matrix);
void printMatrix(vector<double> vector);
void expanded_matrix(vector<vector<double>> &matrix,vector<double> vector);
void row(vector<vector<double>> &matrix, int what,int with_what);
void max_in_column(vector<vector<double>> &matrix,int column);
vector<double> operator-(const vector<double>&a, const vector<double>&b);
vector<double> operator+(const vector<double>&a, const vector<double>&b);
vector<double> operator*(const vector<double>&a, double constanta);
int col_max(vector<vector<double>> &matrix,int col);
int triangulation(vector<vector<double>> &matrix);
double determinant(vector<vector<double>> matrix);
vector<vector<double>> matrix_multiplication(vector<vector<double>> m1,
                                             vector<vector<double>> m2);
void transpose(vector<vector<double>> &matrix);
void scalar_multi(vector<vector<double>> &matrix, double factor);
vector<vector<double>> inverse_matrix(vector<vector<double>> matrix);
double N_1(vector<vector<double>> &matrix);
double N_inf(vector<vector<double>> &matrix);
void Get_matrix(vector<vector<double>> matrix, vector<vector<double>> &tmp,int indRow, int indCol);

int main()
{
    vector<vector<double>> matrix1{{-0.0660,0.6600,1.0050,-6.6900},     //(5, vector<int>(3,0));
                                  {-3.8640,7.2080,47.5740,-292.0740},
                                  {-23.1840,42.9840,285.4450,-1752.1860},
                                  {-3.8640,7.1640,47.5740,-292.0300}};

    vector<double> b{-46.2300,-2219.2580,-13315.8580,-2219.3020};

    expanded_matrix(matrix1,b);

    printMatrix(matrix1);

    vector< vector <double> > matrix;
    QFile file("C:/Users/Миша/Desktop/Проверка.txt");

    setDataToVector(allFileToString(file).split("\n"), matrix);

    printMatrix(matrix);

    size_t number_of_row = 0; //для того, чтобы спускаться по уровням матрицы
                              //например для того, чтобы перейти с матрицы 3x3
                              //на матрицу уровня 2x2 и т.д.

    while (number_of_row < matrix.size())
    {
        max_in_column(matrix,number_of_row);

        double a_ii = (double)matrix[number_of_row][number_of_row];

        for(size_t column=number_of_row; column < matrix[number_of_row].size();column++)
        {
            matrix[number_of_row][column] = (double)matrix[number_of_row][column]/a_ii;//нормировка текущей строки
//            double a_i1 = (double)matrix[row][number_of_row];
//            for(size_t column =number_of_row;column<matrix[row].size();column++)
//            {
//                matrix[row][column] = (double)matrix[row][column]/a_i1;
//            }
        }

        for(size_t row =(number_of_row+1); row < matrix.size();row++)
        {
            double divider = -1*matrix[row][number_of_row];
            vector<double>temp = operator*(matrix[number_of_row],divider);
            matrix[row] = operator+(matrix[row],temp);
        }
        number_of_row++;
    }

    printMatrix(matrix);

    //обратный ход

    vector<double> answer(matrix.size()); //задаем вектор ответов размера количества строк в матрице.

    //k_i = (b_i - sum( от j = k+1 до n = matrix.size())(a_ij*k_j))

    double sum;
    double product; //произведение коэффициента матрицы на число вектора ответов

    for(int k = (matrix.size()-1); k >= 0;k--) //индексы обхода матрицы
    {
        sum = 0; //каждый раз обнуляем сумму
        for (size_t j =k+1; j <= (matrix.size()-1);j++)
        {
            product = matrix[k][j] * answer[j];
            sum = sum + product;
        }
        answer[k] = (matrix[k][(matrix[k].size()-1)] - sum);    //matrix[k][k];
    }

    printMatrix(answer);

    return 0;
}

void printMatrix(vector<vector<double>> matrix)
{
    for(auto i=matrix.begin();i != matrix.end();i++)
    {
        for(auto j=i->begin();j != i->end();j++)
        {
            cout << setw(20) << *j;
        }
        cout << endl;
    }
    cout << endl;
}

void printMatrix(vector<double> vector)
{
    for (auto it = vector.begin(); it!=vector.end();it++)
    {
        cout << scientific <<setprecision(15) <<setw(30) << *it << endl;
    }
    cout << endl;
}

void expanded_matrix(vector<vector<double>> &matrix, vector<double> vector)
{
    for(size_t it=0; it != vector.size();it++)
    {
        matrix[it].push_back(vector[it]);
    }
}

void row_exchange(vector<vector<double>> &matrix, int what,int with_what)
{
    vector<double> tmp = matrix[what];
    matrix[what] = matrix[with_what];
    matrix[with_what] = tmp;

//    int on_what = 0;
//    double max = matrix[0][0];
//    for (int i =0;i!=matrix.size();i++)
//    {
//        if(matrix[i][0]>max)
//        {
//            max = matrix[i][0];
//            on_what = i;
//        }
//    }
}

void max_in_column(vector<vector<double>> &matrix,int column)
{
    double max = matrix[column][column];
    int row_index = column;
    for (size_t row=column;row!=matrix.size();row++)
    {
        if(fabs(matrix[row][column])>fabs(max))
        {
            max = matrix[row][column];
            row_index = row;
        }
    }
    row_exchange(matrix,column,row_index);
}

vector<double> operator-(const vector<double>&a, const vector<double>&b)
{
    vector<double> c(a.size());
    for(size_t i = 0; i < a.size(); ++i)
        c[i] = a[i] - b[i];
    return c;
}

vector<double> operator+(const vector<double>&a, const vector<double>&b)
{
    vector<double> c(a.size());
    for(size_t i = 0; i < a.size(); ++i)
        c[i] = a[i] + b[i];
    return c;
}

vector<double> operator*(const vector<double>&a, double constanta)
{
    vector<double> c(a.size());
    for(size_t i = 0; i < a.size(); ++i)
        c[i] = a[i]*constanta;
    return c;
}

int col_max(vector<vector<double>> &matrix,int col)
{
    int n = matrix.size();
    double max = abs(matrix[col][col]);
    int maxPos = col;
    for (int i = col + 1; i < n; ++i)
    {
        double element = abs(matrix[i][col]);
        if (element > max){
            max = element;
            maxPos = i;
        }
    }
    return maxPos;
}

int triangulation(vector<vector<double>> &matrix)
{
    int n = matrix.size();
    unsigned int swapCount = 0;

    const int num_cols = matrix[0].size();
    for (size_t i = 0; i < n-1; ++i){
        unsigned int imax = col_max(matrix, i);
        if(i != imax){
            swap(matrix[i],matrix[imax]);
            ++swapCount;
        }
        for (int j = i+1; j < n;++j){
            double mul = -matrix[j][i]/matrix[i][i];
            for (int k = i; k<num_cols;++k){
                matrix[j][k] += matrix[i][k]*mul;
            }
        }
    }
    return swapCount;
}

double determinant(vector<vector<double>> matrix)
{
    unsigned int swapCount = triangulation(matrix);
    double determinant = 1;

    if (swapCount % 2 ==1)
        determinant = -1;
    for (size_t i=0;i<matrix.size();++i)
    {
        determinant *= matrix[i][i];
    }
    return determinant;
}

vector<vector<double>> matrix_multiplication(vector<vector<double>> m1,
                                             vector<vector<double>> m2)
{
    vector<vector<double>> answer;
    answer.resize(m1.size());

    for (size_t i=0;i != answer.size();i++)
    {
        answer[i].resize(m2[0].size());
    }

    for(size_t i = 0; i < m1.size();i++)
    {
        for(size_t j = 0; j < m2[0].size();j++)
        {
            answer[i][j] = 0;
            for(size_t k = 0; k<m1[0].size();k++)
            {
                answer[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
    return answer;
}

void transpose(vector<vector<double>> &matrix)
{
    vector<vector<double>> copy = matrix;

    matrix.resize(copy[0].size());

    for(size_t i = 0;i<copy[0].size();i++)
    {
        matrix[i].resize(copy.size());
    }

    for(size_t row = 0; row<copy.size();row++)
    {
        for(size_t col = 0;col<copy[0].size();col++)
        {
            matrix[col][row] = copy[row][col];
        }
    }
}

void scalar_multi(vector<vector<double>> &matrix, double factor)
{
    for(size_t row = 0; row<matrix.size();row++)
    {
        for(size_t col = 0;col<matrix[0].size();col++)
        {
            matrix[row][col] *= factor;
        }
    }
}

double N_1(vector<vector<double>> &matrix)
{
    double N1 = 0;
    for(size_t row = 0; row<matrix[0].size();row++)
    {
        double colsum =0;
        for(size_t col = 0;col<matrix.size();col++)
        {
            colsum += abs(matrix[col][row]);
        }
        N1 = max(colsum,N1);
    }
    return N1;
}

double N_inf(vector<vector<double>> &matrix)
{
    double Ninf = 0;
    for(size_t row = 0; row<matrix.size();row++)
    {
        double rowsum =0;
        for(size_t col = 0;col<matrix[0].size();col++)
        {
            rowsum += abs(matrix[row][col]);
        }
        Ninf = max(rowsum,Ninf);
    }
    return Ninf;
}

vector<vector<double>> inverse_matrix(vector<vector<double>> matrix)
{
    vector<vector<double>> answer;
    answer.resize(matrix.size());
    for(size_t i =0;i<answer.size();i++)
    {
        answer[i].resize(matrix[0].size());
    }

    vector<vector<double>> tmp;

    double det = determinant(matrix);

    for(size_t row = 0; row<matrix.size();row++)
    {
        for(size_t col = 0;col<matrix[0].size();col++)
        {
            Get_matrix(matrix,tmp,row,col);
            answer[row][col] = ((pow(-1.0,(row+col+2))*determinant(tmp))/det);
        }
    }
    transpose(answer);
    return answer;
}

void sum(vector<vector<double>> &matrix1,vector<vector<double>> matrix2)
{
    for(size_t row = 0; row < matrix1.size(); row++)
            for(size_t col = 0; col < matrix1[0].size(); col++)
                matrix1[row][col] += matrix2[row][col];
}

void Get_matrix(vector<vector<double>> matrix, vector<vector<double>> &tmp,int indRow, int indCol)
{
    tmp.resize(matrix.size()-1);

    for(size_t i = 0;i<tmp.size();i++)
    {
        tmp[i].resize(matrix[0].size()-1);
    }

    int ki = 0;
    for (int row = 0; row < matrix.size(); row++){
        if(row != indRow){
            for (int col = 0, kj = 0; col < matrix.size(); col++){
                if (col != indCol){
                    tmp[ki][kj] = matrix[row][col];
                    kj++;
                }
            }
            ki++;
        }
    }
}
