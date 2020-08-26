
#ifndef __GRADIENT_ASCENT__
#define __GRADIENT_ASCENT__

class Number
{
private:
    int ival;
    float fval;

    int i_inc;
    float f_inc;
    
    Number();       // disallow

public:
    Number( int val, int inc );
    Number( float val, float inc );

    Number& operator++();
    Number& operator--();

    // const int operator()();
    // const float operator()();
};

#endif