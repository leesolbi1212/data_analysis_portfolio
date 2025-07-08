# objectdata.py
# 객체데이터 라이브러리

# pickle
import pickle
obj = {
    'name': '홍길동',
    'age': 20
}

# obj.obj파일을 바이너리쓰기모드로 생성해서 객체를 파일에 씀
with open('obj.obj', 'wb') as f:
    pickle.dump(obj, f)

# obj.obj파일을 바이너리읽기모드로 읽어서 화면에 출력
with open('obj.obj', 'rb') as f:
    print(pickle.load(f))

# shelve
import shelve

# key, value를 shelve 파일에 저장하는 함수
def save(key, value):
    with shelve.open('shelve') as f:
        f[key] = value

# shelve 파일에서 key에 해당하는 value를 가져오는 함수
def get(key):
    with shelve.open('shelve') as f:
        return f[key]

save('number', [1, 2, 3, 4, 5])
save('string', ['a', 'b', 'c'])
print(get('number'))
print(get('string'))

    
    
    
    