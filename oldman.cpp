#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
using namespace std;

//用一個int64代表手上所有的牌,short*4個花色

#define TWO		0x0002
#define THREE		0x0004
#define FOUR		0x0008
#define FIVE		0x0010
#define SIX		0x0020
#define SEVEN		0x0040
#define EIGHT		0x0080
#define NINE		0x0100
#define TEN		0x0200
#define JACK		0x0400
#define QUEEN		0x0800
#define KING		0x1000
#define ACE		0x2000

#define StraightFlush		(9<<27)
#define FourOfAKind		(8<<27)
#define	FullHouse		(7<<27)
#define	Flush			(6<<27)
#define	Straight		(5<<27)
#define	ThreeOfAKind		(4<<27)
#define	TwoPair			(3<<27)
#define	OnePair			(2<<27)
#define	HighCard		(1<<27)

typedef union u64i {
    unsigned long long ull;
    unsigned short us[4];
} u64;

char popcnt[16385];	// 並非所有INTEL的CPU都支援  __popcnt, 所以我們預設了一個陣列, 快速計算該數字有幾個BIT為1

void ipopcnt() {
	int i, j;
	char n;
	for(i=0;i<16385;i++) {
		n=0;
		j=i;
		while(j) {
			if(j&1) n++;
			j>>=1;
		}
		popcnt[i]=n;
	}
}

//0~51
unsigned long long cardBit(int n) {
	u64 r;
	r.ull=0;
	r.us[n/13]=(1<<((n%13)+1));
	return r.ull;
}



unsigned int poker5( u64 s ) {	

	// 超級快速地為傳進的5張牌計算一個德州撲克分數, 分數愈大牌愈大
	// 傳進 s[4] 分別代表4種花色有什麼牌，例如 s[0]=10 表示 SPADE 有 2, 4， 4種花色必需剛好5個BIT為1

	unsigned int ans, f;
	unsigned int a= (unsigned int)(s.us[0]|s.us[1]|s.us[2]|s.us[3]);
	char b= popcnt[a];
	if(b==5) {
		ans=f=0;
		if(s.us[0]==0)f++;
		if(s.us[1]==0)f++;
		if(s.us[2]==0)f++;
		if(s.us[3]==0)f++;
		if(a==0x201e) ans=0x001f;			// A2345 最小的順
		else if(popcnt[a+(((~a)+1)&a)]==1) ans=a;	// 其他順子		<--------- 超神,無迴圈一行找出順?!
		if(f==3) {
			if(ans) return( StraightFlush+ans );	// 同花順
			else return( Flush+a );			// 同花
		} else {
			if(ans) return( Straight+ans );		// 順
			else return( HighCard+a );		// 虎爛
		}
	} else if(b==4) {
		ans=(unsigned int)(s.us[0]^s.us[1]^s.us[2]^s.us[3]);	//	<--------- 這樣就定出3個單張的BIT !!!
		return( OnePair+((ans^a)<<13)+ans );		// 1對
	} else if(b==3) {
		ans=(unsigned int)(s.us[0]^s.us[1]^s.us[2]^s.us[3]);
		if( popcnt[ans] == 1 ) {
			return( TwoPair + ((ans^a)<<13)+ans );	// 2對
		} else {
			return( ThreeOfAKind + (((unsigned int)((s.us[0]&s.us[1])|(s.us[2]&s.us[3])))<<13)+ans );
								// 3條			<--------- 簡單濾出3條BIT的方法
		}
	} else if(b==2) {
		ans=(unsigned int)(s.us[0]&s.us[1]&s.us[2]&s.us[3]);
		if(ans) return( FourOfAKind + (ans<<13) + a );	// 4條
		return( FullHouse + (((unsigned int)(s.us[0]^s.us[1]^s.us[2]^s.us[3]))<<13)+a);	// 葫蘆
	}
	return 0;
}

unsigned int poker6(u64 s, unsigned long long m) {		// 6張選5張, return 最大分
	unsigned long long b;
	unsigned int score=0;
	while( m ) {
		b = ((~m)+1)&m;
		s.ull^=b;
		score=max(score,poker5(s));
		s.ull^=b;
		m^=b;
	}
	return score;
}

unsigned int poker7(u64 s) {		// 7張選5張, return 最大分
	unsigned long long m = s.ull;
	unsigned long long b;
	unsigned int score=0;
	while( m ) {
		b = ((~m)+1)&m;
		m^=b;
		s.ull^=b;
		score=max(score,poker6(s,m));
		s.ull^=b;
	}
	return score;
}



#define RATE_TIMES	1000

int winRate( u64 myCard, u64 tableCard ) {
	int i, j, winCount;
	u64 foundCard, hisCard, t;
	unsigned int myScore, hisScore;

	int n = (int)(popcnt[tableCard.us[0]]+popcnt[tableCard.us[1]]+popcnt[tableCard.us[2]]+popcnt[tableCard.us[3]]);
	winCount=0;
	for(i=0; i<RATE_TIMES; i++) {
		foundCard.ull=myCard.ull+tableCard.ull;
		do {
			hisCard.ull=cardBit(rand()%52);
		} while(hisCard.ull & foundCard.ull);
		foundCard.ull += hisCard.ull;
		do {
			t.ull=cardBit(rand()%52);
		} while(t.ull & foundCard.ull);
		foundCard.ull += t.ull;
		hisCard.ull += t.ull;

		for(j=n; j<5; j++) {
			do {
				t.ull=cardBit(rand()%52);
			} while(t.ull & foundCard.ull);
			foundCard.ull += t.ull;
		}
		foundCard.ull-=hisCard.ull;
		myScore=poker7(foundCard);
		foundCard.ull+=hisCard.ull;
		foundCard.ull-=myCard.ull;
		hisScore=poker7(foundCard);
		if(myScore>hisScore) {
			winCount+=2;
		} else if(myScore==hisScore) {
			winCount++;
		}
	}

	return(winCount*500/RATE_TIMES);	// 0 to 1000
}

int think(u64 myCard, u64 tableCard, int myBet, int hisBet, int pot, int *guess) {
	// myBet, hisBet, pot : 	請傳與「每局每家下注上限」的比例，  myBet, hisBet範圍是0-100, pot範圍是0-200
	// guess : 由本局本次BET之前對方的BET猜測的對方強度, 每手牌開始時請幫設成0!!!!!!! 呼叫think時會被改變
	// return : -1 FOLD, 0 CHECK/CALL, 1-100 BET/RAISE(就是下注後,myBet變成的值與下注限額的百分比)
	int a;

	if(myBet>=95) return 0;
	int n = (int)(popcnt[tableCard.us[0]]+popcnt[tableCard.us[1]]+popcnt[tableCard.us[2]]+popcnt[tableCard.us[3]]);
	if( hisBet > myBet ) {
		(*guess)=(*guess)+(hisBet-myBet)*(n+2)/9;
	}

	int w=winRate(myCard,tableCard);
	
	// w = 1000 - ((1000-w)*(100+(*guess))/100);	// 暫時忽略
	
	w=max(0,w);
	w=min(1000,w);
	// printf("<AI判斷 hisbet:%d mybet:%d pot:%d guess:%d w:%d>",hisBet,myBet,pot,(*guess),w);
	if( myBet >= hisBet ) {
		a = (w-500)/5 - pot/2 - myBet;
		if( a <= 0 ) {
			//if(pot/2<10 && rand()%4==0) a=rand()%20;	// NO_RANDOM
			//else return 0;				// NO_RANDOM
			return 0;
		} else {
			a = a * (n+2)/7;
			//if(pot/2<10 && a>20) a=rand()%10+10;		// NO_RANDOM
		}
		//if((rand()%10)<(5-n)) return 0;	// NO_RANDOM			// 無條件埋伏
		return (min(100,a+myBet));			// raise myBet to 
	} else {	
		if(((w-500)/5) <= (hisBet + pot/2) ) {
			if( (hisBet * 2 + pot) * w > (hisBet - myBet) * 1000) {		// 06/05
				return 0;
			} else {
				return -1;
			}
		} else {
			a = (w-500)/5 - pot/2 - hisBet;
			a = a * (n+2)/7;
			if( a <= 0 ) return 0;
			//if((rand()%10)<(5-n)) return 0;	// NO_RANDOM		// 無條件埋伏
			return (min(100,a+hisBet));		// raise myBet to 
		}
	}
}


extern "C" {

  void InitPokerAI(){
	srand( time(NULL) );
	ipopcnt();
  }

  int goThink(unsigned int myCardh, unsigned int myCardl, unsigned int tableCardh, unsigned int tableCardl, int myBet, int hisBet, int pot, int newHand) {
	static int handGuess=0;
	u64 myCard, tableCard;
	myCard.ull=((unsigned long long)myCardh<<32)+(unsigned long long)myCardl;
	tableCard.ull=((unsigned long long)tableCardh<<32)+(unsigned long long)tableCardl;
	if(newHand == 1) handGuess = 0;
	return think(myCard, tableCard, myBet, hisBet, pot, &handGuess);
  }

  unsigned int score7( unsigned int hi, unsigned int lo ) {
	u64 t;
	t.ull=((unsigned long long)hi<<32)+(unsigned long long)lo;
	return poker7(t);
  }

  int goWinRate( unsigned int myCardh, unsigned int myCardl, unsigned int tableCardh, unsigned int tableCardl ) {
	u64 m,t;
	m.ull=((unsigned long long)myCardh<<32)+(unsigned long long)myCardl;
	t.ull=((unsigned long long)tableCardh<<32)+(unsigned long long)tableCardl;
	return winRate(m,t);
  }
}

// g++ -fPIC -shared -o oldman.so oldman.cpp
