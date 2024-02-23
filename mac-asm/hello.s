// See https://opensource.apple.com/source/xnu/xnu-1504.3.12/bsd/kern/syscalls.master for syscalls

.global _main // tells linker that _main is entrypoint
.align 2 // gotta do it for macos arm64

_main:
    adr X0, introtext
    bl _printf

    adr X0, inputprompt

    bl _printf

    mov X4, #64 // Allocate 64 bytes in in stack
    sub SP, SP, X4

    mov X0, SP // Buffer address
    mov X1, X4 // Buffer size
    bl _get_input // stores size of result in X0

    // Parse first number
    ldrb W19, [SP] // Load first byte from [SP] into register and increment SP
    sub W19, W19, #'0' // Subtract '0' from register to convert from ASCII to integer

    mov X0, #'A'
    bl _print_char

    // Skip space
    add SP, SP, #1 // Increment SP to skip the space

    mov X0, #'B'
    bl _print_char

    // Parse second number
    ldrb W20, [SP], #1 // Load next byte from [SP] into register and increment SP
    sub W20, W20, #'0' // Subtract '0' from register to convert from ASCII to integer

    mov X0, #'C'
    bl _print_char
    add W19, W19, W20 // Add the numbers together

    // Convert the result to ASCII
    add W19, W19, #'0'

    // Print the result
    mov X0, #'B'
    bl _print_char

    add SP, SP, X4 // Deallocate buffer

    b _terminate


_get_input: // Stores size of result in X0
    stp X30, X29, [SP, #-16]!
    mov X19, X0 // Buffer address
    mov X20, X1 // Buffer size

    mov X0, #0 // stdin
    mov X1, X19 // buffer
    mov X2, X20 // buffer size
    mov X16, #3 // syscall read
    svc 0 // invoke syscall

    ldp X30, X29, [SP], #16
    ret


_print_char:
    mov X5, SP // Copy SP to X5
    add X5, X5, #15 // Add 15 to X5
    bic X5, X5, #15 // Clear the last 4 bits of X5 to align to 16 bytes

    sub SP, SP, X5 // Allocate enough space on the stack to ensure alignment

    strb W0, [SP, X5, LSL #0] // Store the character on the stack at the aligned address

    mov X1, SP // Copy SP to X1
    add X1, X1, X5 // Add the adjustment to X1
    mov X0, #1 // stdout
    mov X2, #1 // length should be 1 for a single character
    mov X16, #4 // syscall write
    svc 0 // Make the syscall

    add SP, SP, X5 // Deallocate the adjusted amount of space on the stack
    ret

; _print_char:
;     sub SP, SP, #16 // Allocate 16 bytes on the stack
;     strb W0, [SP] // Store the character on the stack
;     mov X1, SP // Copy SP to X1
;     mov X0, #1 // stdout
;     mov X2, #1 // length should be 1 for a single character
;     mov X16, #4 // syscall write
;     svc 0 // Make the syscall
;     add SP, SP, #16 // Deallocate the 16 bytes on the stack
;     ret


_printf:
    mov X1, X0 // string address to print
    mov X2, #0 // index, start at 0

    _strlen:
        ldrb W3, [X1, X2] // get char at index
        cbz W3, _printf_1 // if null terminator (null byte), jump to print
        add X2, X2, #1 // increment index
        b _strlen // loop

    _printf_1:
        mov X0, #1 // stdout
        mov X16, #4
        svc 0
        ret


_input_echo:
    stp X30, X29, [SP, #-16]!
    mov X4, #256 // Allocate 256 bytes in in stack

    sub SP, SP, X4 
    mov X0, SP // Buffer address
    mov X1, X4 // Buffer size
    bl _get_input // stores size of result in X0

    mov X2, X0
    mov X0, #1 // stdout
    mov X1, SP // Buffer address
    mov X16, #4 // syscall write
    svc 0 // Make the syscall

    add SP, SP, X4 // Deallocate buffer

    ldp X30, X29, [SP], #16
    ret


_terminate:
    mov X0, #0 // return 0
    mov X16, #1 // terminate
    svc 0 // syscall


introtext: .ascii "Welcome to crazy calculator!\n\0"
inputprompt: .ascii "Enter two integers to add: \0"

