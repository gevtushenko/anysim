//
// Created by egi on 5/18/19.
//

#ifndef ANYSIM_COMMON_FUNCS_H
#define ANYSIM_COMMON_FUNCS_H

template<typename... Ts>
inline void cpp_unreferenced (Ts&&...) {}

template <class child_class>
class non_copyable
{
public:
  non_copyable (const non_copyable &) = delete;
  child_class & operator = (const child_class &) = delete;

protected:
  non_copyable () = default;
  ~non_copyable () = default; /// Protected non-virtual destructor
};

#endif //ANYSIM_COMMON_FUNCS_H
