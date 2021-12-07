#ifndef __COMMON_STRING_H__
#define __COMMON_STRING_H__

#include <stdlib.h>
#include <string.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define strncasecmp _strnicmp
#define strcasecmp strcmpi
#endif

inline int stringRemoveDelimiter(char delimiter, const char* string)
{
    int str_start = 0;

    while (string[str_start] == delimiter) {
        str_start++;
    }

    if (str_start >= static_cast<int>(strlen(string))) {
        return 0;
    }

    return str_start;
}

inline bool checkCmdLineFlag(int argc, const char** argv, const char* str_ref)
{
    bool found = false;

    if (argc >= 1) {
        for (int i = 1; i < argc; i++) {
            int str_start = stringRemoveDelimiter('-', argv[i]);
            const char *str_argv = &argv[i][str_start];
            const char *equal_pos = strchr(str_argv, '=');

            int argv_length = static_cast<int>(equal_pos == 0 ? strlen(str_argv) : equal_pos - str_argv);
            int length = static_cast<int>(strlen(str_ref));

            if (length == argv_length && !strncasecmp(str_argv, str_ref, length)) {
                found = true;
                continue;
            }
        }
    }

    return found;
}

inline bool getCmdLineArgumentString(int argc, const char** argv, const char* str_ref, char** str_retval)
{
    bool found = false;

    if (argc >= 1) {
        for (int i = 1; i < argc; i++) {
            int str_start = stringRemoveDelimiter('-', argv[i]);
            char* str_argv = const_cast<char *>(&argv[i][str_start]);
            int length = static_cast<int>(strlen(str_ref));

            if (!strncasecmp(str_argv, str_ref, length)) {
                *str_retval = &str_argv[length + 1];
                found = true;
                continue;
            }
        }
    }

    if (!found)
        *str_retval = NULL;

    return found;
}

inline int getCmdLineArgumentInt(int argc, const char** argv, const char* str_ref)
{
    bool found = false;
    int value = -1;

    if (argc >= 1) {
        for (int i = 1; i < argc; i++) {
            int str_start = stringRemoveDelimiter('-', argv[i]);
            char* str_argv = const_cast<char *>(&argv[i][str_start]);
            int length = static_cast<int>(strlen(str_ref));

            if (!strncasecmp(str_argv, str_ref, length)) {
                if (length + 1 <= static_cast<int>(strlen(str_argv))) {
                    int auto_inc = (str_argv[length] == '=') ? 1 : 0;
                    value = strtol(&str_argv[length + auto_inc], NULL, 10);
                }
                else {
                    value = 0;
                }
            }

            found = true;
            continue;
        }
    }

    if (found)
        return value;
    else
        return 0;
}

#endif