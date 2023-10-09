#pragma once

#include <type_traits>


namespace ck {

template<typename T>
struct TypeIdentity {
    using type = T;
};

template <typename T, typename... Ts>
struct IsOneOf {
    static constexpr bool value = false;
};

template <typename T, typename... Others>
struct IsOneOf<T, T, Others...> {
    static constexpr bool value = true;
};

template <typename T, typename First, typename... Others>
struct IsOneOf<T, First, Others...>: IsOneOf<T, Others...> {};

}

template <template<typename> typename Pred, typename... Ts>
struct AllOf {
    static constexpr bool value = false;
};

template <template<typename> typename Pred, typename First, typename... Others>
struct AllOf<Pred, First, Others...> {
    static constexpr bool value = Pred<First>::value && AllOf<Pred, Others...>::value;
};

template <template<typename> typename Pred, typename T>
struct AllOf<Pred, T> {
    static constexpr bool value = Pred<T>::value;
};
