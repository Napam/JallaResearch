function easter(year) {
    // g - Golden year - 1
    // c - Century
    // h - (23 - Epact) mod 30
    // i - Number of days from March 21 to Paschal Full Moon
    // j - Weekday for PFM (0=Sunday, etc)
    // p - Number of days from March 21 to Sunday on or before PFM
    //     (-6 to 28 methods 1 & 3, to 56 for method 2)
    // e - Extra days to add for method 2 (converting Julian
    //     date to Gregorian date)
    
    const y = year
    const g = y % 19
    const c = Math.floor(y/100)
    const h = (c - Math.floor(c/4) - Math.floor((8*c + 13)/25) + 19*g + 15) % 30
    const i = h - Math.floor(h/28) * (1-Math.floor(h/28)*Math.floor(29/(h+1))*Math.floor((21-g)/11))
    const j = (y+Math.floor(y/4)+i+2-c+Math.floor(c/4)) % 7
    const p = i - j
    const d = 1 + (p + 27 + Math.floor((p+6)/40)) % 31
    const m = 3 + Math.floor((p+26)/30) 

    return new Date(y, m-1, d)
}

console.log(easter(2022))
