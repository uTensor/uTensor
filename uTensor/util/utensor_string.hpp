#include <string.h>

namespace utensor {
    class string{
        private:
            uint32_t value;
            const char* cstr = NULL;

            uint32_t hash(const char* c){
                int v = 7;
                for(int i = 0; i < strlen(c); i++){
                    v = v*31 + c[i];
                    
                }

                return v;
            }

        public:
            string(const char* that){
                value = hash(that);
                cstr = that;
            }
            string(){
                value = hash("");
            }
            // bool operator < (const string& that){ return this->value < that.value; }
            // bool operator == (const string& that){ return this->value == that.value; }
            bool operator < (const string& that) const { return this->value < that.value; }
            bool operator == (const string& that) const { return this->value == that.value; }

            uint32_t get_value() const { return value; }
            const char* c_str() const { return cstr; }
    };

}

namespace std {
    template<> struct hash<utensor::string>
    {
        typedef utensor::string argument_type;
        typedef std::size_t result_type;

        result_type operator()(argument_type const& s) const noexcept
        {
            return s.get_value();
        }
    };
}
