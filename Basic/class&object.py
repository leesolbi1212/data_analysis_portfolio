# class&object.py
# 클래스, 객체

# 클래스 생성
class Human:
    humanCount = 0 # 클래스 변수 (객체들이 값을 공유하는 변수)
    def __init__(self, name, age): # 생성자 : 객체 변수의 값을 초기화 하는 생성자
        self.name = name # self.name : 객체 변수, name: 생성자 파라미터
        self.age = age
    def getAge(self): # 메소드, getter, 객체 변수의 값을 획득
        return self.age
    def setAge(self, age): # 메소드, setter, 객체 변수의 값을 변경
        self.age = age

# 객체 생성
hong = Human('홍길동', 20)
print(hong.humanCount)
print(hong.name)
print(hong.age)

# 객체 생성
kang = Human('강감찬', 30)
print(kang.humanCount)
print(kang.name)
print(kang.age)

# 클래스 변수의 값을 변경
# 클래스 변수는 클래스를 통해 생성된 모든 객체가 값을 공유
Human.humanCount = 1
print(hong.humanCount)
print(kang.humanCount)

# 객체지향의 3대 개념

# 1. 상속 (Inheritance)
#    이미 잘 만들어진 것을 가져다 재사용

# 2. 다형성 (Polymophism)
#    같은 형태인데 다른 성질을 가짐

# 3. 추상화 (Abstraction)
#    본질을 잃어버리지 않는 선에서 최대한 단순화

# 상속 & 오버라이딩
class Vehicle:
    def __init__(self, name, tireCount):
        self.name = name
        self.tireCount = tireCount
    def getName(self):
        return f'이 탈것의 이름은 {self.name} 입니다!'

# Vehicle을 상속받은 Car
class Car(Vehicle):
    def getName(self): # Vehicle의 getName 오버라이딩(overriding, 재정의)
        return f'이 차의 이름은 {self.name} 입니다!'

car = Car("Bentz", 4)
print(car.name, car.tireCount)

car = Vehicle("BMW", 4)
print(car.getName())

car = Car("BMW", 4)
print(car.getName())

# 오버라이딩
class Bird():
    def __init__(self, name, sound):
        self.name = name
        self.sound = sound
    def cry(self):
        print(f'새는 {self.sound} 소리를 냅니다!')

class Eagle(Bird):
    def cry(self):
        print(f'{self.name}는 {self.sound} 소리를 냅니다!')

class Chicken(Bird):
    def cry(self):
        print(f'{self.name}는 {self.sound} 소리를 냅니다!')

eagle = Eagle("독수리", "꽤꽤")
chicken = Chicken("닭", "꼬끼오")
eagle.cry()
chicken.cry()

birdList = [
    Chicken('닭1', '꼬끼오'),
    Chicken('닭2', '꼬끼꼬끼'),
    Eagle('독수리1', '꽤꽤'),
    Eagle('독수리2', '꾸꾸')
]

# 오버라이딩의 목적은 동일한 메소드를 호출하더라도
# 타입에 따라서 다른 기능을 하도록 함
for bird in birdList:
    bird.cry()
















        
        
        
