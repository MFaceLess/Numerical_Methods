#include <QCoreApplication>
#include <iostream>
#include <vector>
#include <math.h>
#include <iomanip>
#include <fstream>
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
vector<double> Gauss(vector<vector<double>> matrix,vector<double> b);
vector<double> Sweep_Method(vector<double> a,
                            vector<double> b,
                            vector<double> c,
                            vector<double> d);
vector<double> matrix_multiplication(vector<vector<double>> m1,vector<double> m2);

int main()
{
    vector< vector <double> > matrix1;

    vector<double> a;
    vector<double> b;
    vector<double> c;
    vector<double> d;

    QFile file("C:/Users/Миша/Desktop/Метод прогонки.txt");

    setDataToVector(allFileToString(file).split("\n"), matrix1);

    //запушим 0 в a1=0
    a.push_back(0);

    //заполнение векторов значениями из файла
    for (size_t row =0; row<matrix1.size();row++)
    {
        for(size_t col =0; col<matrix1[row].size();col++)
        {
            switch (row)
            {
                case 0:
                    a.push_back(matrix1[row][col]);
                    break;
                case 1:
                    b.push_back(matrix1[row][col]);
                    break;
                case 2:
                    c.push_back(matrix1[row][col]);
                    break;
                case 3:
                    d.push_back(matrix1[row][col]);
                    break;
            }
        }
    }
    //запушим 0 в c_n=0
    c.push_back(0);
//-------------------------------------------------------------------------------
    vector<vector<double>> coefficient;
    coefficient.resize(b.size());
    for(int i=0;i<coefficient.size();i++)
    {
        coefficient[i].resize(b.size());
    }
    for(size_t row =0;row<coefficient.size();row++)
    {
        for(size_t col =0;col<coefficient[row].size();col++)
        {
            if (col == row)
            {
                coefficient[row][col] = b[row];
            }
            if (row == col+1)
            {
                coefficient[row][col] = a[row];
            }
            if (row+1 == col)
            {
                coefficient[row][col] = c[row];
            }
        }
    }
//-------------------------------------------------------------------------------
    vector<double> answer;
    answer = Sweep_Method(a,b,c,d);
    cout<<"answer"<<endl;
    printMatrix(answer);
//------------------------------------------------------------------------------
    vector<double> discrepancy1 = operator-(d,matrix_multiplication(coefficient,answer)); // вектор невязки

    cout<<"discrepancy"<<endl;
    printMatrix(discrepancy1);

    double N1_temp = N_1(discrepancy1); //норма 1 невязки

    double Ninf_temp = N_inf(discrepancy1);//норма беск невязки

    cout << "N1:" << N1_temp <<endl <<"Ninf:"<< Ninf_temp<<endl<<endl;
//------------------------------------------------------------------------------
//change several coefficients to 0.01
    for (size_t k =0; k<d.size();k++)
    {
        d[k] = d[k] + 0.01;
    }

    vector<double> stability;
    stability = Sweep_Method(a,b,c,d);
    cout<<"changing answer"<<endl;
    printMatrix(stability);
//-------------------------------------------------------------------------------
//вычисляем устойчивость
    vector<double> discrepancy = operator-(d,matrix_multiplication(coefficient,stability)); // вектор невязки

    cout<<"stability"<<endl;
    printMatrix(discrepancy);

    double N1 = N_1(discrepancy); //норма 1 невязки

    double Ninf = N_inf(discrepancy);//норма беск невязки

    cout << "N1:" << N1 <<endl <<"Ninf:"<< Ninf<<endl<<endl;

    return 0;
}

vector<double> Sweep_Method(vector<double> a,
                            vector<double> b,
                            vector<double> c,
                            vector<double> d)
{
    vector<double> answer;

    bool is_valid = false;
    vector<double> not_valid;
    not_valid.push_back(0);

    //Проверяем достаточное условие применимости метода прогонки
    for(size_t k = 0;k<b.size();k++)
    {
        if (abs(b[k])>(abs(a[k])+abs(c[k])))
        {
            is_valid = true;
        }
    }

    for(size_t k = 0;k<b.size();k++)
    {
        if ((abs(b[k])<(abs(a[k])+abs(c[k]))) ||(is_valid == false))
        {
            cout << "Sufficient condition is not met";
            return not_valid;
        }
    }

    cout<<"Sufficient condition is met, let's go!"<<endl;

    //Прямая прогонка
    vector<double> alpha;
    vector<double> betta;
    vector<double> gamma;

    //первую итерацию при  k=0 делаем вручную
    gamma.push_back(b[0]);
    alpha.push_back(-c[0]/gamma[0]);
    betta.push_back(d[0]/gamma[0]);

    for(size_t k = 1;k<(b.size()-1);k++)
    {
        gamma.push_back(b[k]+a[k]*alpha[k-1]);
        alpha.push_back(-c[k]/gamma[k]);
        betta.push_back((d[k]-a[k]*betta[k-1])/gamma[k]);
    }
    //проводим итерацию для k=n(n=(b.size()-1), т.к индексирование идет с 0)
    gamma.push_back(b[b.size()-1]+a[b.size()-1]*alpha[b.size()-2]);
    betta.push_back((d[b.size()-1]-a[b.size()-1]*betta[b.size()-2])/gamma[b.size()-1]);

//--------------------------------------------------------------------------------------------
    //Обратная прогонка
    //для n-ой итерации
    answer.push_back(betta[betta.size()-1]);

    //для k-ой итерации
    for(int k = betta.size()-2;k>-1;k--)
    {
        answer.push_back(betta[k]+alpha[k]*answer[betta.size()-2-k]);
    }

    //инвертирование вектора ответов
    for(size_t ind = 0;ind < answer.size()/2;ind++)
    {
        swap(answer[ind],answer[answer.size()-1-ind]);
    }
//----------------------------------------------------------------------------------------
    return answer;
}

//возращает вектор ответов методом Гаусса
vector<double> Gauss(vector<vector<double>> matrix,vector<double> b)
{
    expanded_matrix(matrix,b);

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
