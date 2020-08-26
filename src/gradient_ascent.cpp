
#include "gradient_ascent.hpp"

#include <vector>

Number::Number( int val, int inc )
{
    ival = val;
    fval = 0.0f;
    i_inc = inc;
    f_inc = 0.0f;
}
    

Number::Number( float val, float inc )
{
    ival = 0;
    fval = val;
    i_inc = 0;
    f_inc = inc;
}


Number& Number::operator++()
{
    if ( ival )
    {
        ival += i_inc;
    }
    else
    {
        fval += f_inc;
    }
    return *this;
}


Number& Number::operator--()
{
    if ( ival )
    {
        ival -= i_inc;
    }
    else
    {
        fval -= f_inc;
    }
    return *this;
}

/*
const int Number::operator()()
{
    return ival;
}

const float Number::operator()()
{
    rturn fval;
}
*/



void grad_ascent( const std::vector< const Number& >& curr_params, const Number& last_val,
                  std::vector< Number& >& new_params, const Number& curr_val )
{

}
