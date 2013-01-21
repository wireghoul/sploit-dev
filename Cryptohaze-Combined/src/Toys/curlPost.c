// Testing CURL posting to URLs and getting response.

#include <stdio.h>
#include <string.h>
#include <curl/curl.h>

int main(void) {
    CURL *curl;
    CURLcode res;

    static const char *postthis = "name=bitweasil|testing&value=foobar";

    // structs for our post
    struct curl_httppost* post = NULL;
    struct curl_httppost* last = NULL;
    
    char stringToSend[10] = "test\0test";
    
    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "http://localhost/testing/test.php");
        //curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postthis);

        //curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long) strlen(postthis));

        // Add us some binary data... maybe?
 curl_formadd(&post, &last, 
              CURLFORM_COPYNAME, "rawHashUpload", 
              CURLFORM_BUFFER, "rawHashUpload", 
              CURLFORM_BUFFERPTR, stringToSend, 
              CURLFORM_BUFFERLENGTH, (long)9, 
              CURLFORM_END); 
 
        
         curl_formadd(&post, &last, CURLFORM_COPYNAME, "rawHashSubmit",
              CURLFORM_COPYCONTENTS, "1", CURLFORM_END);
        
         curl_formadd(&post, &last, CURLFORM_COPYNAME, "rawHashLength",
              CURLFORM_COPYCONTENTS, "16", CURLFORM_END);

         curl_easy_setopt(curl, CURLOPT_HTTPPOST, post);
        
        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            printf("curl error: %s\n", curl_easy_strerror(res));
        }

        /* always cleanup */
        curl_easy_cleanup(curl);
    }
    return 0;
}


