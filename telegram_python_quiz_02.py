a = [1, 2, 3]
a = tuple(a)
a[0] = 2

print(a)

# Tuples are used when data must not change
# tuple(a) creates (1, 2, 3)
# tuples do not allow assignment
# So this line is illegal: a[0] = 2, then Error