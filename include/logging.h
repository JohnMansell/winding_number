
#ifndef LOGGING_H
#define LOGGING_H


#include <stdio.h>
#include <stdarg.h>
#include <printf.h>
#include <string.h>
#include <stdbool.h>

// --- System
#include <sys/ioctl.h>
#include <sys/signal.h>
#include <execinfo.h>
#include <unistd.h>


/* ===============================================================
                        Logging Macros
================================================================== */

typedef enum {
    RESET = -1,
    DEBUG = 0,
    INFO,
    WARNING,
    ERROR,
    CRITICAL,
    ALWAYS,
    SUCCESS,
} LoggerLevel;


// --- Shorten File Name
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

/**
 * -------------- DEBUG ONLY -------------
 *      Grab File Name + Line Number each time
 */
#ifdef __DEBUG__
    #define __logDebug(...)     console_log(DEBUG,   __FILENAME__, __LINE__, __VA_ARGS__)
    #define __logInfo(...)      console_log(INFO,    __FILENAME__, __LINE__, __VA_ARGS__)
#else
    #define __logDebug(...)
    #define __logInfo(...)
#endif

// ------------ DEBUG + RELEASE -----------
    #define __logWarning(...)   console_log(WARNING, __FILENAME__, __LINE__, __VA_ARGS__)
    #define __logError(...)     console_log(ERROR,   __FILENAME__, __LINE__, __VA_ARGS__)
    #define __logCritical(...)  console_log(CRITICAL,__FILENAME__, __LINE__, __VA_ARGS__)
    #define __logAlways(...)    console_log(ALWAYS,  __FILENAME__, __LINE__, __VA_ARGS__)
    #define __logSuccess(...)   console_log(SUCCESS, __FILENAME__, __LINE__, __VA_ARGS__)
    #define __logTest(...)      log_test(__FILENAME__, __LINE__, __VA_ARGS__)

// ---- Line Break
#define __logLineBreak(...) log_line_break(__FILENAME__, __LINE__, __VA_ARGS__)

// --- Sig Fault Handler
#define fatal_error(message) fatal_error_helper( __FILENAME__, __LINE__, (message));
// void segv_handler(int sig);

#ifdef __cplusplus
extern "C"
{
#endif  // __cplusplus
void segv_action(int sig, siginfo_t *info, void *ucontext);

/* ===============================================================
                        Logging Functions
================================================================== */
void set_terminal_color(LoggerLevel level);
void set_terminal_message_color(LoggerLevel level);

void fatal_error_helper(const char* calling_file, int line, const char* message, ...);

void log_line_break(const char* calling_file, int line, const char* title, ...);
void log_test(const char *calling_file, int line, bool condition, const char *fmt_message, ...);
void console_log(LoggerLevel level, const char *calling_file, const int line, const char *fmt_message, ...);

/* ---------------------------------------------------------------------------------
        Shorten File Name
        Alternative when using CMAKE
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$(subst
            ${CMAKE_SOURCE_DIR}/,,$(abspath $<))\"'")
------------------------------------------------------------------------------------ */

/* ---------------------------------------------------------------------------------
    Logging Example
    https://tuttlem.github.io/2012/12/08/simple-logging-in-c.html
--------------------------------------------------------------------------------- */


#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // LOGGING_H
