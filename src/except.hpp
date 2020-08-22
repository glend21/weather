/*
 * Exception class
 */

 #include <exception>

class RainException : std:exception
{
protected:
    std::string text;

public:
    RainException( const std::string& msg ) noexcept
    {
        this->text = msg;
    }

    const std::string& getMsg() const
    {
        return this->text;
    }
};