#ifdef __cplusplus
extern "C" {
#endif

//===============================================================================================================================================================================================================200
//	DESCRIPTION
//===============================================================================================================================================================================================================200

// Returns:	0 if string does not represent integer
//			1 if string represents integer

//===============================================================================================================================================================================================================200
//	NUM CODE
//===============================================================================================================================================================================================================200

//======================================================================================================================================================150
//	ISINTEGER FUNCTION
//======================================================================================================================================================150

int isInteger(char *str){

	//====================================================================================================100
	//	make sure it's not empty
	//====================================================================================================100

	if (*str == '\0'){
		return 0;
	}

	//====================================================================================================100
	//	if any digit is not a number, return false
	//====================================================================================================100

	for(; *str != '\0'; str++){
		if (*str < 48 || *str > 57){	// digit characters (need to include . if checking for float)
			return 0;
		}
	}

	//====================================================================================================100
	//	it got past all my checks so I think it's a number
	//====================================================================================================100

	return 1;
}

//===============================================================================================================================================================================================================200
//	END NUM CODE
//===============================================================================================================================================================================================================200

#ifdef __cplusplus
}
#endif
