### Question 1
(a) It returns '\x00'.

(b) The character's string representation is more for programmers while its printed representation is for everyone.

(c) For the string representation, it displays as '\x00'. For its printed representation, it does not appear, as chr(0) references the null character.

### Question 2
(a) UTF-8 encoding represents unicode characters ranging from 1-4 bytes, with the most common characters being represented as 1 byte. UTF-16 encoding represents unicode characters ranging from 2-4 bytes. UTF-32 characters represent UTF-32 characters as 4 bytes. It is preferable to training our tokenizer on UTF-8 encoded bytes because on average, it will be shorter in length. This is because the more common characters are more likely to appear in the training text, and since more common unicode characters are only represented with 1 byte, the UTF-8 encoded byte representation will be much shorter than the UTF-16 or UTF-32 representation.

(b) This function is incorrect because characters may be represented by multiple bytes in UTF-8 encoding, meaning if you decode assuming each byte maps cleanly to one character, you will encounter an error. An example input string for which decode_utf8_bytes_to_str_wrong produces incorrect output is "π". 

(b) bytes([192, 128]) does not decode to any Unicode character(s). This is because not every 2 byte sequence maps to a valid Unicode character.