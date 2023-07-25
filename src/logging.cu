#include "logging.h"

// #include <algorithm>
// #include <iostream>
// #include <iterator>
// #include <vector>

// --- System
#include <sys/ioctl.h>
#include <sys/signal.h>
#include <execinfo.h>
#include <unistd.h>

// --- C
#include <cstdio>
#include <iostream>
#include <cstring>


// --- CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>

#define BT_BUF_SIZE (300U)



typedef enum {

    black  = 0,
    red    = 1,
    green  = 2,
    yellow = 3,
    blue   = 4,
    purple = 5,
    cyan   = 6,
    white  = 7,
    reset  = 9,
} TerminalColor;


typedef enum {
    style_none = 0,
    bold  = 1,
    underline = 4,
    inverse = 7
} TerminalStyle;

const char* Logger_Level_Names[7] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "ALWAYS", "SUCCESS"};

void set_terminal_color(LoggerLevel level)
{
    // --- Only Print Colors on TTY terminal
    if (!isatty(fileno(stderr)))
        return;

    TerminalColor background = reset;
    TerminalColor text_color = reset;
    TerminalStyle style = style_none;

    switch (level)
    {
    case (WARNING):
        background = yellow;
        text_color = black;
        style = bold;
        break;

    case (ERROR):
        background = red;
        text_color = white;
        break;

    case (CRITICAL):
        background = purple;
        text_color = black;
        style = bold;
        break;

    case (ALWAYS):
        background = blue;
        text_color = black;
        break;

    case (DEBUG):
        background = green;
        text_color = black;
        break;

    case (INFO):
        background = white;
        text_color = black;
        break;

    case (SUCCESS):
        background = cyan;
        text_color = black;

    default:
        break;
    }

    fprintf(stderr, "\033[%d;3%d;4%dm", style, text_color, background);
    fflush(stderr);
}

void set_terminal_message_color(LoggerLevel level)
{
    // --- Only Print Colors on TTY terminal
    if (!isatty(fileno(stderr)))
        return;

    TerminalColor background = reset;
    TerminalColor text_color = reset;
    TerminalStyle style = style_none;

    switch (level)
    {
    case (WARNING):
        text_color = yellow;
        style = bold;
        break;

    case (ERROR):
        text_color = red;
        break;

    case (CRITICAL):
        text_color = purple;
        style = bold;
        break;

    case (ALWAYS):
        text_color = blue;
        break;

    case (DEBUG):
        text_color = green;
        break;

    case (INFO):
        text_color = white;
        break;

    case (SUCCESS):
        text_color = cyan;
        break;

    default:
        break;
    }

    fprintf(stderr, "\033[%d;3%d;4%dm", style, text_color, background);
    fflush(stderr);
}

void console_log(LoggerLevel level, char const *calling_file, const int line, char const *fmt_message, ...)
{
    // --- Initialize Log Line
    set_terminal_message_color(level);
    size_t space = strlen(Logger_Level_Names[level])/2;
    fprintf(stderr, " ::%*s%*s :: %21s [%4d] ---> ", 4+space, Logger_Level_Names[level], 4-space, "", calling_file, line);

    // --- Print Variable Message Arguments
    set_terminal_color(level);
    va_list arguments;
    va_start(arguments, fmt_message);
    vfprintf(stderr, fmt_message, arguments);
    va_end(arguments);

    // --- Clean Up
    set_terminal_color(RESET);
    fprintf(stderr, "\n");
}


void log_test(const char *calling_file, const int line, bool condition, const char *fmt_message, ...)
{

    // ToDo :: Make this a templated function and pass __va_args__ to console logger.
    //  This is duplicate code.

    LoggerLevel level = condition ? SUCCESS : ERROR;

    // --- Initialize Log Line
    set_terminal_message_color(level);
    size_t space = strlen(Logger_Level_Names[level])/2;
    fprintf(stderr, " ::%*s%*s :: %27s [%4d] ---> ", 4+space, Logger_Level_Names[level], 4-space, "", calling_file, line);

    // --- Print Variable Message Arguments
    set_terminal_color(level);
    va_list arguments;
    va_start(arguments, fmt_message);
    vfprintf(stderr, fmt_message, arguments);
    va_end(arguments);

    // --- Clean Up
    set_terminal_color(RESET);
    fprintf(stderr, "\n");
}


void __log_line_break(const char* calling_file, const int line, const char* title, ...)
{
    // This function will output the following text to terminal :

    /* ----------------------------
    *       < format title >
    *  ----------------------------
     */

    // --- Terminal Window Size
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

    // --- Dashed Line
    char dashed_line[w.ws_col];
    memset(dashed_line, '-', sizeof(dashed_line)-1);
    dashed_line[w.ws_col - 1] = '\0';

    // --- Top Line
    set_terminal_message_color(INFO);
    fprintf(stderr, "%s\n", dashed_line);
    fprintf(stderr, "\t\t\t\t\t\t\t\t");

    // --- Variable Message Arguments
    va_list arguments;
    va_start(arguments, title);
    vfprintf(stderr, title, arguments);
    va_end(arguments);

    // --- Bottom Line
    fprintf(stderr, "\n%s\n", dashed_line);

    // --- Clean Up
    set_terminal_color(RESET);
    fprintf(stderr, "\n");
}


void fatal_error_helper(char const *calling_file, int line, char const *fmt_message)
{
    fflush(stdout); // don't cross the streams
    __logError(calling_file, line, fmt_message);
    exit(EXIT_FAILURE);
}


void segv_action(int sig, siginfo_t *info, void *ucontext)
{

    __logCritical("============ >> SEG FAULT << =================");

    (void) sig;
    // ucontext_t* ctx = reinterpret_cast<ucontext_t*>(ucontext);

    // ------------------------------------------------------
    //              GPU SEG FAULT
    // ------------------------------------------------------
    //    cudaError_t result_ = cudaGetLastError();
    //    cudaError_t run_result_ = cudaPeekAtLastError();
    //
    //    if(result_ != cudaSuccess)
    //    __logError("Cuda Last Error = %s : %s", cudaGetErrorName(result_), cudaGetErrorString(result_));
    //
    //    if(run_result_ != cudaSuccess)
    //    __logError("CUDA RUN Error  = %s : %s", cudaGetErrorName(run_result_), cudaGetErrorString(run_result_));
    //

    // ------------------------------------------------------
    //              Back Trace
    // ------------------------------------------------------

    // --- Initialize Variables
    int nptrs;
    void *buffer[BT_BUF_SIZE];
    char **strings;

    // --- Get Backtrace Info
    nptrs = backtrace(buffer, BT_BUF_SIZE);

    strings = backtrace_symbols(buffer, nptrs);
    if (strings == nullptr) {
        __logCritical("No backtrace symbols");
        exit(EXIT_FAILURE);
    }

    for (int j=0; j<nptrs; j++)
    {
        char *p_open, *p_close, *p_slash, *p_plus;
        p_open = strchr(strings[j], '(');
        p_close = strchr(strings[j], ')');

        char address[64] = {'0'};
        char calling_file[64] = {'0'};
        char func_name[32] = {'0'};

        if ( p_open != NULL)
        {

            p_slash = strrchr(strings[j], '/');
            if (p_slash == NULL)
                continue;

            p_plus = strchr(p_open, '+');
            if (p_plus == NULL)
                continue;

            strncpy(address, p_plus, (p_close - p_plus));
            strncpy(calling_file, p_slash+1, (p_open - p_slash)-1);
            strncpy(func_name, p_open+1, (p_plus - p_open));

            // console_log(CRITICAL, calling_file, 999, "%s %s", func_name, strings[j]);
        }

        /* find first occurence of '(' or ' ' in message[i] and assume
        * everything before that is the file name. (Don't go beyond 0 though
        * (string terminator)*/
        int p = 0;
        while( (strings[j][p] != '(') &&
               (strings[j][p] != ' ') &&
               (strings[j][p] != 0) )
            ++p;

        char syscom[256] = {'0'};

        __logDebug("Strings J = %s", strings[j]);
        printf("\n ----- First -----\n");
        sprintf(syscom,"addr2line %s -e %.*s -C -f", address, p, strings[j]);
        fflush(stdout);
        system(syscom);

        printf("\n----- Second ------\n");
        sprintf(syscom,"addr2line %s -e %.*s -C -f -j .text", address, p, strings[j]);
        fflush(stdout);
        system(syscom);
    }

    backtrace_symbols_fd(buffer, nptrs, STDOUT_FILENO);
    //    free(strings);

    abort();
}

