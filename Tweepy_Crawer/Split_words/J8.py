#encoding=utf-8
import jieba
import operator
import jieba.analyse
import time
jieba.set_dictionary('dict.txt')#繁體中文
jieba.load_userdict("userdict.txt")#法1 :自定義語料庫

def Split_Sentence(sentence):
	spl_word = []
	words = jieba.cut(sentence, cut_all=False)
	#print ("Output 精確模式 Full Mode：")
	for word in words:
	    spl_word.append(word)
	return spl_word

def TF_IDF(sentence):
	tags = jieba.analyse.extract_tags(sentence, topK=10)#Tf-Idf
	print(",".join(tags))
	return tags

def Make_Score(word1,word2):
	Count_Score = 0
	for i in word1:
		Score = 0
		for j in word2:
			if i in j:
				print(i)
				Score = 1
		Count_Score += Score
	return Count_Score

if __name__ == '__main__':
	target = "支持陳前總統保外就醫"
	sentence = "前總統陳水扁律師團今天說，法務部若依醫療鑑定小組認定，准扁保外就醫，重視醫療人權，在國際能見度，絕對有正面的意義。外界關切陳水扁能否獲准保外就醫，陳水扁律師團成員石宜琳今天發聲明指出，陳水扁從羈押到服刑已逾6年1個多月，被關這麼久已經夠了。聲明指出，扁在獄中罹患重度憂鬱症、重度阻塞型睡眠呼吸暫止症、神經退化性疾病，嚴重口吃、手不斷顫抖，攝護腺肥大合併中度排尿功能障礙，一天甚至漏尿失禁逾80次，扁真的前總統陳水扁律師團今天說，法務部若依醫療鑑定小組認定，准扁保外就醫，重視醫療人權，在國際能見度，絕對有正面的意義。外界關切陳水扁能否獲准保外就醫，陳水扁律師團成員石宜琳今天發聲明指出，陳水扁從羈押到服刑已逾6年1個多月，被關這麼久已經夠了。聲明指出，扁在獄中罹患重度憂鬱症、重度阻塞型睡眠呼吸暫止症、神經退化性疾病，嚴重口吃、手不斷顫抖，攝護腺肥大合併中度排尿功能障礙，一天甚至漏尿失禁逾80次，扁真的病了，早就應該儘快「保外就醫」，這是受刑人的醫療基本人權，法治重視人權國家不也都是如此嗎？聲明表示，如果不是專業醫師，如果沒有親眼看到扁現在病況，就不應該任意批評、任意揣測「阿扁是裝的」。這次台中榮總15人醫療鑑定小組，鑑定結果一致認定通過「阿扁應保外就醫，居家治療」，難道這會是假的嗎？聲明稿指出，如果法務部依照15人醫療鑑定小組一致認定通過「阿扁應返家治療」，而准予保外就醫，相信這對監獄「醫療人權」的重視，在國際能見度上，絕對有正面的意義，也應該有加分的作用。聲明表示，也期待法務部主管當局，對於其他在監服刑的受刑人，有類似罹患重大疾病，而有必要保外就醫的人，也能同等重視，讓獄政基本人權邁一大步。1031230"
	target_imp = Split_Sentence(target)
	sentence_spl = Split_Sentence(sentence)
	'''
	print ("Input："+sentence)
	start = time.time()
	Split_Sentence(sentence)
	#jieba.suggest_freq("能見度",True)#法2 :可以調節單字詞的詞頻，使其能被分出來
	now = time.time()
	print("Time : "+str(now-start))'''
	Score = Make_Score(target_imp,sentence_spl)
	print("This Score is "+str(Score))