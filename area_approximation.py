size = 0.01

count = 0
x = -1 + size/2
while x < 1:
    y = -1 + size/2
    while y < 1:
        if x*x + y*y <= 1:
            count += 1
        y += size
    x += size
area = count * size * size
print(area)
