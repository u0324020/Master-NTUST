#encoding=utf-8
import jieba
import operator
import jieba.analyse
from snownlp import normal
from snownlp import SnowNLP
from snownlp import seg
from snownlp.summary import textrank
from scipy import rank
import time
jieba.set_dictionary('dict.txt')#繁體中文
jieba.load_userdict("userdict.txt")#法1 :自定義語料庫

def Snow_rank(sentence):#snownlp斷字
	sents = normal.get_sentences(sentence)#斷句
	word = []
	import_sent = []
	for sent in sents:
		words = seg.seg(sent)#斷詞
		words = normal.filter_stop(words)#刪除停用詞
		word.append(words)
	rank = textrank.TextRank(word)#句子重要度排名
	rank.solve()
	for index in rank.top_index(5):
		import_sent.append(sents[index])
	'''keyword_rank = textrank.KeywordTextRank(word)#字重要度排名
	keyword_rank.solve()
	for w in keyword_rank.top_index(5):
		print(w)'''
	return import_sent
def Sentence2Words(sentence):#jieba斷字
	spl_word = []
	words = jieba.cut(sentence, cut_all=False)
	#print ("Output 精確模式 Full Mode：")
	for word in words:
		spl_word.append(word)
	return spl_word

def Emotion(sentence):
	text = sentence
	sents = normal.get_sentences(text)
	emotion_score = 0
	text_count = 0
	for sent in sents:
		s = SnowNLP(sent)
		score = s.sentiments
		emotion_score += score
		text_count += 1
		#print(sent)
		#print(score) #positive 的機率
	#print("Avg Score is "+str(emotion_score/text_count))
	return str(emotion_score/text_count)

def TF_IDF(sentence):
	tags = jieba.analyse.extract_tags(sentence, topK=10)#Tf-Idf
	print(",".join(tags))
	return tags

def Calculate_Score(word1,word2):
	Count_Score = 0
	Score = 0
	for i in word2:
		if i in word1:
			#print(i)
			Score += 1
	Count_Score += Score
	return Count_Score

def Calculate_Snow_Score(word1,word2):
	Count_Score = 0
	Score = 0
	for i in word1:
		for j in word2:
			if i in j:
				#print(i)
				Score = 1
		Count_Score += Score
	return Count_Score
def Calculate_Main(target,sentence):
	target_imp = Sentence2Words(target)
	sentence_spl = Sentence2Words(sentence)
	sent_emotion_score = Emotion(sentence)
	tar_emotion_score = Emotion(target)
	sent_rank = Snow_rank(sentence)
	#TF_IDF(sentence)
	Score = Calculate_Score(target,sentence_spl)
	Snow_Score = Calculate_Snow_Score(target_imp,sent_rank)
	Remotion_Score = 0
	for i in sent_rank:
		S = Emotion(i)
		Remotion_Score+=float(S)
	Avg_Remontion = Remotion_Score/5
	print("============================================================")
	print("Target Emotion score = "+ tar_emotion_score)
	print("Full Sentence Emotion score = "+ sent_emotion_score)
	print("Top 5 sentence Emotion score = "+str(Avg_Remontion))
	print("Compare keywords score = "+str(Score))
	print("Compare Snow Rank keywords score = "+str(Snow_Score))
	E_score = (float(sent_emotion_score)+float(Avg_Remontion))*0.5-float(tar_emotion_score)
	W_score = (float(Score)+float(Snow_Score))
	return (E_score)*50+W_score



if __name__ == '__main__':
	target = "支持陳前總統保外就醫"
	sentence = "前總統陳水扁律師團今天說，法務部若依醫療鑑定小組認定，准扁保外就醫，重視醫療人權，在國際能見度，絕對有正面的意義。外界關切陳水扁能否獲准保外就醫，陳水扁律師團成員石宜琳今天發聲明指出，陳水扁從羈押到服刑已逾6年1個多月，被關這麼久已經夠了。聲明指出，扁在獄中罹患重度憂鬱症、重度阻塞型睡眠呼吸暫止症、神經退化性疾病，嚴重口吃、手不斷顫抖，攝護腺肥大合併中度排尿功能障礙，一天甚至漏尿失禁逾80次，扁真的前總統陳水扁律師團今天說，法務部若依醫療鑑定小組認定，准扁保外就醫，重視醫療人權，在國際能見度，絕對有正面的意義。外界關切陳水扁能否獲准保外就醫，陳水扁律師團成員石宜琳今天發聲明指出，陳水扁從羈押到服刑已逾6年1個多月，被關這麼久已經夠了。聲明指出，扁在獄中罹患重度憂鬱症、重度阻塞型睡眠呼吸暫止症、神經退化性疾病，嚴重口吃、手不斷顫抖，攝護腺肥大合併中度排尿功能障礙，一天甚至漏尿失禁逾80次，扁真的病了，早就應該儘快「保外就醫」，這是受刑人的醫療基本人權，法治重視人權國家不也都是如此嗎？聲明表示，如果不是專業醫師，如果沒有親眼看到扁現在病況，就不應該任意批評、任意揣測「阿扁是裝的」。這次台中榮總15人醫療鑑定小組，鑑定結果一致認定通過「阿扁應保外就醫，居家治療」，難道這會是假的嗎？聲明稿指出，如果法務部依照15人醫療鑑定小組一致認定通過「阿扁應返家治療」，而准予保外就醫，相信這對監獄「醫療人權」的重視，在國際能見度上，絕對有正面的意義，也應該有加分的作用。聲明表示，也期待法務部主管當局，對於其他在監服刑的受刑人，有類似罹患重大疾病，而有必要保外就醫的人，也能同等重視，讓獄政基本人權邁一大步。1031230"
	sentence2 = "在藍營身陷貪瀆疑雲之際，民進黨主席蘇貞昌昨日證實，「我也連署了爭取阿扁保外就醫連署」，並強調民進黨很關心扁的健康，希望社會各界能分進合擊，共同達成目標。對此，扁的兒子陳致中回應，相當感謝蘇貞昌的幫忙，未來扁家將與民進黨一起努力，讓父親盡早離開監獄，以取得良好的醫療照顧。台大醫師柯文哲、前北社社長陳昭姿日前發起救扁連署行動後，日前赴民進黨中央邀請蘇貞昌參與連署，儘管蘇未在第一時間簽名，但事後包括前在藍營身陷貪瀆疑雲之際，民進黨主席蘇貞昌昨日證實，「我也連署了爭取阿扁保外就醫連署」，並強調民進黨很關心扁的健康，希望社會各界能分進合擊，共同達成目標。對此，扁的兒子陳致中回應，相當感謝蘇貞昌的幫忙，未來扁家將與民進黨一起努力，讓父親盡早離開監獄，以取得良好的醫療照顧。台大醫師柯文哲、前北社社長陳昭姿日前發起救扁連署行動後，日前赴民進黨中央邀請蘇貞昌參與連署，儘管蘇未在第一時間簽名，但事後包括前副總統呂秀蓮、前行政院長游錫?、謝長廷、前民進黨主席蔡英文等人，都接連參與連署，因而引發挺扁人士對蘇的不解。但從前天起，四十多個親綠社團，及全台二六六位民進黨籍市議員，陸續接到蘇貞昌的公開信，內容則指出，基於人道考量，民進黨已協調各級民代聲援扁的醫療人權，並聯繫意見領袖發揮影響力，盡早促成此事，「本人亦已於日前連署，在此，希冀閣下以醫療人權為本，以社會和諧為念，發揮影響力共同參與連署，為台灣化解對立。」不僅如此，昨日下午民進黨更召開中常會，正式敲定黨中央支持保外就醫行動的立場。會後，蘇貞昌主動接受媒體訪問，強調他已按照日前中執會的決議，親自行文給民進黨各級民意代表，希望各級議會能提案籲請馬政府讓扁保外就醫。對此，扁辦回應，扁長期遭遇上銬羞辱、精神凌虐、高度隔離、不人道對待，已造成嚴重心理創傷、重度憂鬱，「是故，陳前總統必需離開監獄」，這才是最有效的治療方法。如今，在有了民進黨的支持後，相信馬政府「頑石亦能點頭」，終會同意保外就醫。"
	sentence3 = "近來酒醉肇事的事件頻傳，尤其「葉少爺」事件更引發輿論撻伐，也有立委主張將刑罰加重，甚或學習鄰近的日本，將酒醉駕車致死者的法定刑，提高至20年的修法建議。惟在立法例不同下，如何沿襲，卻是個須審慎面對的課題。日本關於酒醉駕車的刑事處罰，規定於《道路交通法》中，其中第65條第1項，即明文一律禁止酒後駕車的行為，若帶有酒氣駕駛遭取締，依同法第117條之2-2第1款，只要呼氣酒精濃度達於每公升0.25毫克，不管有無酒醉，即可處以3年以下有期徒刑或50萬日幣以下的罰金；而若達於所謂酒醉，即不能安全駕駛的狀態，則依據同法第117條之2第1款，可處5年以下有期徒刑或1百萬日幣以下罰金。而為了更有效防止酒駕，日本《道路交通法》甚至規定，對酒駕者為車輛或酒類提供，甚或容認其開車者，亦須為連帶處罰，而將道德義務提升至法律層次。惟日本如此嚴格且嚴密的刑罰規定，似乎遏止不了酒駕事件不斷攀升，而在2000年，更創下交通事故死傷人數超過百萬的最高峰。因此，在2001年，為解決此問題，日本國會即於《刑法》中，增訂第208條之2，其中的第1項即明文，受到酒精或藥物影響，致難使所駕駛的交通工具正常運作而仍行駛者，若因此致人於傷，可處15年以下有期徒刑，若因此造成他人死亡，更可處1年以上20年以下有期徒刑。因此罪的法定刑上限，已與故意傷害罪相當，為了避免紊亂故意與過失的區別，此酒駕肇事罪乃被置於刑法分則的故意傷害罪章中，這也代表，此種極為嚴重的酒駕致人於死傷的行為，已脫離過失而進入故意犯的領域。惟如此的特殊立法，卻也有諸多問題存在。因將酒醉肇事當成是一個故意行為，顯然紊亂了《刑法》中，關於故意與過失的區別，尤其是日本故意殺人罪，其法定刑乃從5年起跳，酒駕致死最高到20年，顯然已違反罪刑相當。同時，在1年到20年如此寬廣的空間下，實難防止因人、因案件而異的差別對待，且法官為了避免輕重失衡，是否敢於重判，亦成問題。更何況，由於列入故意犯之故，關於證據與事實的認定，必然趨於嚴格，被告更可能強力抗辯，而使審判趨於漫長，則期待重刑而能嚇阻酒駕的效果，必因此遞減。所以，在我國目前的酒醉駕車肇事的法定刑為1到7年，若再為加重，必為10年或以上的情況下，亦會出現相同的問題。所以，日本重罰政策，絕非可以模仿的對象，反倒是關於日本《道路交通法》中，針對單純的酒醉駕車的處罰基準，則可為參考的對象。因我國《刑法》第185條之3第1項，關於酒駕須達於不能安全駕駛的判斷，法院對於警察以酒精濃度為標準的作法，往往不以為然，卻又未能理出一套客觀基準，其結果必陷入司法的恣意認定，則日本法直接以一定酒精濃度為處罰標準，或可成為減少爭議的取法對象。總之，刑罰要有一般預防的效果，其前提在有效率的訴追，也因此，第一線執法的警察，強力、密集且有效率的取締，才是嚇阻酒駕最有效的手段。若欲寄望加重刑罰即可來為解決，不僅是種緣木求魚的想法，更無助於問題的解決。作者為真理大學法律系副教授"
	Score_Before_Rank=[]
	Score_Before_Rank.append(Calculate_Main(target,sentence))
	Score_Before_Rank.append(Calculate_Main(target,sentence2))
	Score_Before_Rank.append(Calculate_Main(target,sentence3))
	print("============================================================")
	print(Score_Before_Rank)
	Score_Before_Rank.sort()
	print(Score_Before_Rank)