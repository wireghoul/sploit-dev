#include <sstream>
#include <stdint.h>
#include <cstdlib>
#include <time.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <iostream>


//g++ -I/usr/local/include/boost/ test.cpp -o test -lboost_serialization

    typedef struct HashPlain {

	    	    friend class boost::serialization::access;
	            std::vector<uint8_t> hash; /**< Hash in file order */
		    std::vector<uint8_t> password; /**< Password related to the hash, or null */
		    char passwordPrinted; /**< True if the password has been printed to screen */
		    char passwordFound; /**< True if the password is found. */
		    char passwordOutputToFile; /**< True if the password has been placed in the output file. */
		   
		    template <class Archive> 
	    	    void serialize(Archive &ar, const unsigned int version)
			{
				ar & hash & password;
				ar & passwordPrinted & passwordFound & passwordOutputToFile;
			}		
		    } HashPlain;

class hashPlainContainter
{
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar & myVector;	
	}
	
	std::vector<HashPlain> myVector;
  	
	void printVector(std::vector<uint8_t> vec)
	{
		std::vector<uint8_t>::iterator i;
		for(i = vec.begin(); i < vec.end(); i++)
			std::cout<<" "<<*i;
		std::cout<<std::endl;
	}	
	
	protected: 
		hashPlainContainter(const HashPlain hashp)
		{
			myVector.push_back(hashp);
		}
	public:
		void addItemToVector(HashPlain item)
		{
			myVector.push_back(item);	
		}

		void printContents(){
			std::vector<HashPlain>::iterator i;
			int j = 0;
			for (i=myVector.begin(); i < myVector.end(); i++)
			{
				std::cout<<"--------------"<<std::endl;
				std::cout<<"	hash:";
				printVector(i->hash);
				std::cout<<"	password:";
				printVector(i->password);
				std::cout<<"	passwordPrinted: "<<i->passwordPrinted;
				std::cout<<"	passwordFound: "<<i->passwordFound;
				std::cout<<"	passwordOutputToFile: "<<i->passwordOutputToFile;
				std::cout<<std::endl;
				j++;
			}
		}
		
		hashPlainContainter(){
		}
		virtual ~hashPlainContainter(){}
};


int main()
{
	hashPlainContainter test, test2;
	//construct sample HashPlains
	std::srand(std::time(NULL));
	int vectorLength = std::rand()%10+1;

	std::vector<uint8_t> vec; //used as a temporary variable to fill HashPlains with sample data
	for(int j=0;j<std::rand()%10+1;j++) //Random number of HashPlains
	{
		HashPlain sample;
		for(uint8_t i=48; i < 48+vectorLength; i++)
			vec.push_back(std::rand() % 10 + i);
		sample.hash = vec;
		vec.clear();
		for(uint8_t k=65; k < 65+vectorLength; k++)
			vec.push_back(std::rand() % 10 + k);
		sample.password = vec;
		vec.clear();
		sample.passwordPrinted = '0';
		sample.passwordFound = '1';
		sample.passwordOutputToFile = '0';

		test.addItemToVector(sample);
	}


	
	std::cout<<"Contents before serialization."<<std::endl;
	test.printContents();
	
	
	std::cout<<"Serializing"<<std::endl;
	std::stringstream fs(std::stringstream::in| std::stringstream::out);
	{
		boost::archive::text_oarchive oa(fs);
		oa << test;
	}
	
	std::cout<<"Deserializing"<<std::endl;
	{
		boost::archive::text_iarchive ia(fs);
		ia >> test2;
	}
	std::cout<<"Contents after serialization"<<std::endl;
	test2.printContents();

	return 0;}
