# module&package.py
# 모듈 : 변수와 함수들을 모아놓은 파일
# 패키지 : 모듈들을 모아놓기 위한 폴더 (네임스페이스)

# 모듈 임포트
# module.calc패키지에 있는 calc모듈을 c로 알리어싱(별칭)
import module.calc.calc as c
print(c.add(3, 5))

# 함수 임포트
# module.calc패키지에 있는 calc모듈에서 add함수와 multi함수 임포트
from module.calc.calc import add, multi
print(add(3, 5))
print(multi(3, 5))


