#pragma once
#include <vector>
#include <cmath>
#include <string>

//-------These are the base structs-----------
struct vector3 {
    double x, y, z;
};

struct atom {
    int Z;
    vector3 R;
};

struct primitive_gaussian {
    double alpha;
    double coeff;
    double norm;
};

struct basis_function {
    vector3 center;

    // angular momentum
    int lx, ly, lz;

    std::vector<primitive_gaussian> prims;
};

//-------------------------------

struct basis_info{

    std::vector<int> index;
    std::vector<int> shells;

    std::vector<int> lx;
    std::vector<int> ly;
    std::vector<int> lz;

    std::vector<double> cx;
    std::vector<double> cy;
    std::vector<double> cz;

    std::vector<std::string> types;
    std::vector<std::string> atoms;
};

struct scf_data{

    std::vector<double> density;
    std::vector<double> coefficient;
    std::vector<double> mo_energy;

    double E;
    double duration;
    std::string start;
    std::string end;
    int iterations;
    bool converged;

    std::vector<std::pair<std::string, 
    vector3>> system;

    basis_info basis;
    std::string hardware;
    std::string BasisSet;
    int charge;
    int K;

};


struct uhf_data{

    std::vector<double> Pa;
    std::vector<double> Pb;
    std::vector<double> Ca;
    std::vector<double> Cb;
    std::vector<double> Ea;
    std::vector<double> Eb;

    double E_nuc;
    double E_ele;
    double E_tot;

    double duration;
    std::string start;
    std::string end;
    int iterations;
    bool converged;

    std::vector<std::pair<std::string, 
    vector3>> system;

    basis_info basis;
    std::string hardware;
    std::string BasisSet;
    int charge;
    int K;
    int Na, Nb;

};


struct cube_data{

    std::vector<double> field;
    vector3 origin;
    double dx;
    int res;
    
};

struct square{

    double L;
    vector3 center, A, B;

    // defines a square plane in 3D space
    // where C is center and CA is parallel 
    // to one of the sides

};

//--------------------------------------------------------------------------------
//some basic function to work with those stucts easier

inline
vector3 position_vector(double x1, double x2, double x3){
    vector3 r;
    r.x = x1;
    r.y = x2;
    r.z = x3;

    return r;
}

inline
double distance2(const vector3& r1, const vector3& r2){

    return (r1.x-r2.x)*(r1.x-r2.x) + 
           (r1.y-r2.y)*(r1.y-r2.y) + 
           (r1.z-r2.z)*(r1.z-r2.z);

}

inline
double distance(const vector3& r1, const vector3& r2){
    return std::sqrt(distance2(r1, r2));
}

inline
double deg2rad(double theta){
    return theta * M_PI / 180.00;
}

inline
vector3 polar3d(double r, double theta, double phi){
    vector3 position;
    position.x = r * sin(theta) * cos(phi);
    position.y = r * sin(theta) * sin(phi);
    position.z = r * cos(theta);

    return position;
}


inline
vector3 v3add(vector3 r1, vector3 r2){
    vector3 sum;
    sum.x = r1.x + r2.x;
    sum.y = r1.y + r2.y;
    sum.z = r1.z + r2.z;

    return sum;
}

inline
vector3 v3scale(vector3 r, double a){
    vector3 r1;
    r1.x = a * r.x;
    r1.y = a * r.y;
    r1.z = a * r.z;

    return r1;
}

inline
vector3 v3sub(vector3 r1, vector3 r2){

    return v3add(r1, v3scale(r2, -1.0));

}

inline
double v3dot(vector3 r1, vector3 r2){

    return r1.x * r2.x +
           r1.y * r2.y +
           r1.z * r2.z;
}

inline
vector3 v3cross(vector3 r1, vector3 r2){
    vector3 r3;

    r3.x = r1.y * r2.z - r2.y * r1.z;
    r3.y = -r1.x * r2.z + r2.x * r1.z;
    r3.z = r1.x * r2.y - r2.x * r1.y;

    return r3;
}

inline
double v3mod2(vector3 r){

    return r.x * r.x + 
           r.y * r.y + 
           r.z * r.z;

}

inline 
double v3norm(vector3 r){

    return std::sqrt(v3mod2(r));

}

inline
vector3 v3normalize(vector3 r){

    return v3scale(r, 1 / v3norm(r));

}
