from collections import Counter
import pandas as pd
import numpy as np
import re
from ckonlpy.tag import Twitter
# from hanspell import spell_checker
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

custom_stopwords = [
    "범죄도시", "범죄 도시", "범죄", "도시", "죽지",
    "늑대사냥", "늑대", "사냥",
    "공조", "인터네셔날","인터내셔날",
    "한산", "용", "용의", "출현", "용의출현", "장군님", "장군",
    "뜨거운피", "피", "뜨거운",
    "외계인", "외계", "인",
    "영화", "무비", "영화관", "한국영화",
    "진짜", "진자", "완전", "존나", "졸라", "느낌", "뭔가", "제일", "개", "보고", "사람", "보지",
    "어요", "오늘", "그냥", "생각", "면서", "더니", "인적", "거지", "보기", "나름", "살짝",
    "정말", "대박", "역대", "최고", "어제", "편이", "계속", "요소", "처럼", "이나", "역시", "부분", "던데",
    "스포", "개봉", "한번", "내내", "구나", "때문", "어서", "정도", "다가", "다시", "누가", "덕분", "항상",
    "봤는데", "왔는데",
    "스포일러가 포함된 감상평입니다."
]

#  ckonlpy에 추가할 단어 지정
custom_noun = [

    #  [ 범죄도시 ]
    "장이수", "강해상", "손석구", "마석도", "마동석", "마블리", "박지환", "윤계상",
    "2부", "2탄", "2편", "1부", "1탄", "1편",
    #  * 같은 단어 : 장이수=이수, 마석도=마동석=마블리
    #  * 불용어 처리하면 안되는 단어 :
    #  - 구씨 : 배우 손석구가 다른 작품에서 맡았던 케릭터 이름

    #  [ 늑대사냥 ]
    "서인국", "장동윤", "박호산", "정소민", "고창석", "성동일",
    "콘에어", "콘 에어", "잔인",
    #  * 같은 단어 : 콘에어=콘 에어
    #  * 불용어 처리하면 안되는 단어 :
    #  - 콘에어 : 니콜라스 케이지 출연 1997년 영화. 늑대사냥과 비슷한 소재를 다루고있음.

    #  [ 공조2 ]
    "현빈", "유해진", "윤아", "다니엘 헤니", "다니엘헤니", "다니엘", "헤니", "진선규",
    "장영남", "박훈", "임성재", "림청렬", "청렬", "강진태", "진태", "장명준", "박소연",
    "박상위", "박민영",
    "불시착", "사랑의 불시착",
    # 같은 단어 : 다니엘헤니=다니엘=헤니, 불시착=사랑의 불시착

    #  [ 외계인1부 ]
    "류준열", "김우빈", "김태리", "소지섭", "염정아", "조우진", "이하늬",
    "무륵", "김현중", "이안", "문도석", "흑설", "청운",
    "2부", "2탄", "2편", "1부", "1탄", "1편", "돈", "SF", "sf", "SF물", "sf물",
    #  * 불용어 처리하면 안되는 단어 :
    #  - 돈 : 한 글자 '돈 아깝다'같은 말이 종종 보임.
    #  - SF, sf : 장르명. 한글이 아니라고해서 없애면 안됨.

    #  [ 한산 ]
    "박해일", "안성기", "변요한", "손현주", "김성규", "김성균", "김향기", "택연", "옥택연",
    "이순신", "장군님", "장군", "어영담", "원균", "준산", "히데요시", "도요토미", "도요토미 히데요시",
    "한산도", "대첩", "한산도 대첩", "임진왜란", "유키나가", "학익진",
    #  * 같은 단어 : 이순신=이순신장군
    #  * 불용어 처리하면 안되는 단어 :
    #  - 하라 : '전군 출정하라', '선회하라', '발포하라'가 이 영화의 명대사인가봄.

    # 뜨거운피 배우명/등장인물명
    "정우", "김갑수", "최무성", "지승현", "김해곤", "윤지혜", "이홍내", "정호빈"
    #  * 불용어 처리하면 안되는 단어 :
    #  - 짱구 : 배우 정우가 다른 작품에서 맡았던 케릭터 이름

    ]

### 전처리 모듈
def preproc(reviews, custom_stopwords, custom_noun, spellcheck=False) :
    print("전처리 시작")
    #  문서군 변수 생성
    documents = []

    #  기존 불용어 위에 사용자 지정 불용어 적용
    with open("./data/한국어불용어_new.txt", "r", encoding="utf-8") as f :
        stop_words = f.read()
    f.close()
    stop_words = stop_words.split(",")
    stop_words.extend(custom_stopwords)

    #  사용자 지정 단어 사전 추가
    custom_okt = Twitter()
    for n in custom_noun:
        custom_okt.add_dictionary(n, "Noun")

    #  결측치가 존재하는 record 제거
    reviews.dropna(inplace=True)

    #  의미있는한글(ㅋㅋ, ㅎㅎ이런거 말고), 알파벳, 숫자, 띄어쓰기 제외한 글자 삭제
    for idx, comment in enumerate(reviews["comment"]) :
        comment = re.sub(r"[^가-힣a-zA-Z0-9 ]", "", comment)
        reviews.iloc[idx,2] = comment

    #  띄어쓰기, 맞춤법 자동 교정 (주의!! 엄청 오래걸림!!!)
    # if spellcheck == True :
    #     for comment in tqdm(reviews["comment"]) :
    #         comment = spell_checker.check(comment).checked
    #         reviews.iloc[idx,2] = comment

    #  불용어 제거, 한글자 단어 제거
    for comment in reviews["comment"] :
        document = ""
        words = custom_okt.nouns(comment)
        for word in words :
            if (len(word) >= 2) or (word in custom_noun) :
                if word not in stop_words :
                    document += word + " "
        document = document.rstrip()
        documents.append(document)

    print("전처리 완료, 문서군 생성\n")
    #  문서군 변수 리턴
    return documents

def cnt(documents, filename="", cnt_th=10) :
    print("빈도수 기반 키워드 추출 시작")
    #  문서군 내의 모든 문서들을 하나의 문서로 모음
    documents_sum = []
    for document in documents :
        words = document.split(" ")
        if words[0] != "" :   # 공백이 카운트되는것 방지용
            documents_sum.append(words)
    documents_sum = sum(documents_sum, [])

    #  단어들의 빈도수 세기
    counter = Counter(documents_sum)
    vocab_sorted = sorted(counter.items(),  key=lambda x : x[1], reverse=True)

    #  빈도수 cnt_th 이상인 단어만 추출
    vocab_result = []
    for vocab in vocab_sorted :
        if vocab[1] >= cnt_th :
            vocab_result.append(vocab)

    #  변수 vocab_result를 데이터프레임으로 만들기
    data = np.array([None for _ in range(len(vocab_result)*2)])
    data = data.reshape((len(vocab_result),2))
    result_df = pd.DataFrame(data, columns=["noun", "count"])

    for idx, vocab in enumerate(vocab_result) :
        result_df.iloc[idx, 0] = vocab[0]
        result_df.iloc[idx, 1] = vocab[1]

    #  지정한 파일명으로 csv 파일 저장
    #  (파일명을 지정하지 않으면 파일로 저장되지 않고 리턴만 함.)
    if filename != "" :
        result_df.to_csv("{}.csv".format(filename), index=False, sep=",", encoding="utf-8")
        print("csv파일 저장 완료")

    print("빈도수 기반 키워드 추출 완료\n")
    return result_df


df = pd.read_csv('./data/naver_movie.csv')
# sample = df.loc[:10]
a = preproc(df, custom_stopwords, custom_noun)
print(a)
b = cnt(a)
c = b.iloc[0,1]
print(b.head())
print(c)