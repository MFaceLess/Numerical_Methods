#include <QCoreApplication>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <vector>
#include <QTextStream>
#include <QFile>
#include <QDebug>
#include <QTextCodec>
#include <qtextcodec.h>

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
    QString temp = in.readAll();
    aFile.close();
    return temp;
}

void setDataToVector(const QStringList &aStringList,
                     vector< vector <double> > &aVector)
{
    for (size_t row = 0;row < (aStringList.size());row++)
    {
        QStringList stroka = aStringList.at(row).split(" ");
        vector<double> simple;
        for (size_t i =0;i<stroka.size();i++)
        {
            if ((stroka[i] != "") && (stroka[i] != "\t") && (stroka[i] != "@"))
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

vector<double> matrix_multiplication(vector<vector<double>> m1,vector<double> m2);
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
double N_1(vector<double> &matrix);
double N_inf(vector<double> &matrix);
vector<double> Gauss(vector<vector<double>> &matrix,vector<double> b,vector<double> x);
vector<double> Jacobi_method(vector<vector<double>> matrix,vector<double> b,vector<double> x);
vector<double> Seidel_method(vector<vector<double>> matrix,vector<double> b,vector<double> x);

int main()
{
    vector< vector <double> > matrix1;

    vector<double> b1_temp;

    vector<double> x1_temp;

    QFile file("C:/Users/Миша/Desktop/Хорошо обусловленная матрица.txt");

    setDataToVector(allFileToString(file).split("\n"), matrix1);

    for (size_t row =0; row<matrix1.size();row++)
    {
        x1_temp.push_back(matrix1[row][matrix1[row].size()-1]);
        matrix1[row].pop_back();
        b1_temp.push_back(matrix1[row][matrix1[row].size()-1]);
        matrix1[row].pop_back();
    }

    vector<double> answer;

    answer = Seidel_method(matrix1,b1_temp,x1_temp);

    printMatrix(answer);
}

vector<double> Seidel_method(vector<vector<double>> matrix,vector<double> b,vector<double> x)
{
    double EPS = 0.0000001; //задаем точность
    vector<vector<double>> answer;

    //инициализируем матрицу бетта и вектор c

    vector<vector<double>> betta;
    betta.resize(matrix.size());

    for(size_t i = 0;i<betta.size();i++)
    {
        betta[i].resize(matrix[i].size());
    }

    vector<double> c;
    c.resize(b.size());

    for(size_t row=0;row<matrix.size();row++)
    {
        for(size_t col=0;col<matrix[row].size();col++)
        {
            c[row] = b[row]/matrix[row][row];
            //-----------------------------------------------------
            if (row == col)
            {
                betta[row][col] = 0;
            } else
            {
                betta[row][col] = -matrix[row][col]/matrix[row][row];
            }
        }
    }

    if(N_1(betta)>=1)
    {
        cout<<"||B||>=1"<<endl;
        return {0};
    }
    //Проиниализируем матрицы B1 и B2
    //--------------------------------------------------------------------
    vector<vector<double>> B1;
    vector<vector<double>> B2;

    B1.resize(betta.size());
    for(size_t i = 0;i<B1.size();i++)
    {
        B1[i].resize(betta[i].size());
    }

    B2.resize(betta.size());
    for(size_t i = 0;i<B2.size();i++)
    {
        B2[i].resize(betta[i].size());
    }

    for(size_t row=0;row<B1.size();row++)
    {
        for(size_t col=0;col<B1[row].size();col++)
        {
            if (col > row)
            {
                B2[row][col] = betta[row][col];
                B1[row][col] = 0;
            }
            if (col < row)
            {
                B1[row][col] = betta[row][col];
                B2[row][col] = 0;
            }
            if (col == row)
            {
                B1[row][col] = betta[row][col];
                B2[row][col] = betta[row][col];
            }
        }
    }
    //B1 - нижняя треуголная матрица
    //B2 - верхняя треугольная матрица
    //--------------------------------------------------------------------
    //находим EPS1
    double EPS1 = ((1-N_1(betta))*EPS)/N_1(B2);

    //задаем начальное приближение
    vector<double> temp;
    temp.resize(b.size());
    for(size_t col=0;col<temp.size();col++)
    {
        temp[col] = 5;
    }

    answer.push_back(temp);

    vector<double> difference;
    vector<double> sum;
    sum.resize(temp.size());

    do
    {
        for(size_t row=0;row<temp.size();row++)
        {
            sum.clear();
            sum.resize(temp.size());
            for(size_t col=0;col<betta[row].size();col++)
            {
                sum[row] += betta[row][col]*temp[col];
            }
            temp[row] = sum[row]+c[row];
        }
        answer.push_back(temp);

        difference = answer[answer.size()-1] - answer[answer.size()-2];
    }while(N_1(difference) >= EPS1);

    return answer[answer.size()-1];
}

vector<double> Jacobi_method(vector<vector<double>> matrix,vector<double> b,vector<double> x)
{
    double EPS = 0.0000001; //задаем точность
    vector<vector<double>> answer;

    //инициализируем матрицу бетта и вектор c

    vector<vector<double>> betta;
    betta.resize(matrix.size());

    for(size_t i = 0;i<betta.size();i++)
    {
        betta[i].resize(matrix[i].size());
    }

    vector<double> c;
    c.resize(b.size());

    for(size_t row=0;row<matrix.size();row++)
    {
        for(size_t col=0;col<matrix[row].size();col++)
        {
            c[row] = b[row]/matrix[row][row];
            //-----------------------------------------------------
            if (row == col)
            {
                betta[row][col] = 0;
            } else
            {
                betta[row][col] = -matrix[row][col]/matrix[row][row];
            }
        }
    }

    if(N_1(betta)>=1)
    {
        cout<<"||B||>=1"<<endl;
        return {0};
    }
    //находим EPS1
    double EPS1 = ((1-N_1(betta))*EPS)/N_1(betta);

    //задаем начальное приближение
    vector<double> temp;
    temp.resize(b.size());
    for(size_t col=0;col<temp.size();col++)
    {
        temp[col] = 5;
    }

    answer.push_back(temp);
    vector<double> difference;

    size_t row =0;
    do
    {
        temp = matrix_multiplication(betta,answer[row])+c;
        answer.push_back(temp);
        row++;

        difference = answer[row] - answer[row-1];
    }while(N_1(difference) >= EPS1);

    return answer[row];
}

vector<double> Gauss(vector<vector<double>> &matrix,vector<double> b,vector<double> x)
{
    vector<vector<double>> copy = matrix;

    expanded_matrix(matrix,b);

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

    vector<double> discrepancy = operator-(x,answer); // вектор невязки

    printMatrix(discrepancy);

    double N1 = N_1(discrepancy); //норма 1 невязки

    double Ninf = N_inf(discrepancy);//норма беск невязки

    vector<vector<double>> obratka;

    obratka = inverse_matrix(copy);

    double norma1_obr_matrix = N_1(obratka);

    double norma_inf_obr_matrix = N_inf(obratka);

    double absolute_error_1 = norma1_obr_matrix*N1;

    double absolute_error_2 = norma_inf_obr_matrix*Ninf;

    cout <<"Unit Residual Norm and 3 residual "<< endl <<"1:"<< N1 <<endl<<"inf:"<<Ninf<<endl<<endl;

    cout <<"Absolute_errors"<<endl<<"1:"<< absolute_error_1 << endl <<"inf:"<<absolute_error_2 << endl<<endl;

    double relative1_error = absolute_error_1/N_1(x);

    double relative2_error = absolute_error_2/N_inf(x);

    cout <<"Relative_errors"<<endl<<"1:"<< relative1_error << endl <<"inf:"<<relative2_error << endl<<endl;

    cout <<"inverse_matrix:"<<endl;
    printMatrix(inverse_matrix(copy));

    double cond1 = N_1(obratka)*N_1(copy);

    double cond_inf = N_inf(obratka)*N_inf(copy);

    cout << "cond1:" << endl << cond1<< endl <<"cond_inf:"<<endl << cond_inf<<endl<<endl;

    return answer;
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
        cout << scientific <<setprecision(10) <<setw(20) << *it << endl;
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

double N_1(vector<double> &matrix)
{
    double N1 = 0;
    for(size_t row = 0; row<matrix.size();row++)
    {
        N1 += abs(matrix[row]);
    }
    return N1;
}

double N_inf(vector<double> &matrix)
{
    double Ninf;
    for(size_t row = 0; row<matrix.size();row++)
    {
        Ninf = abs(matrix[0]);
        if (abs(matrix[row]) > Ninf)
        {
            Ninf = abs(matrix[row]);
        }
    }
    return Ninf;
}

vector<double> matrix_multiplication(vector<vector<double>> m1,vector<double> m2)
{
    vector<double> answer;
    answer.resize(m1.size());
    for(size_t row=0; row<m1.size(); row++)
    {
        answer[row]=0;
        for(size_t col=0; col<m1[row].size(); col++)
        {
            answer[row]+=m1[row][col]*m2[col];
        }
    }
    return answer;
}
