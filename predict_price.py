# 파일 열기
with open('theta.csv', 'r') as file:
    lines = file.readlines()

theta = [float(i) for i in lines[0].split(',')]

input_km = float(input('Enter the mileage in kilometers: '))
print(f"Estimated price for that mileage: {theta[0] + theta[1] * input_km}")
