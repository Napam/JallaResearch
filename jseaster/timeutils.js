
/**
 * Based on PYthon's dateutil easter implementation
 * @param {number} year 
 * @returns {object}
 */
function calcEasterSunday(year) {
  if (typeof year !== 'number') {
    throw new Error('Year must specified as a number')
  }

  const y = year
  const g = y % 19
  const c = Math.floor(y / 100)
  const h = (c - Math.floor(c / 4) - Math.floor((8 * c + 13) / 25) + 19 * g + 15) % 30
  const i = h - Math.floor(h / 28) * (1 - Math.floor(h / 28) * Math.floor(29 / (h + 1)) * Math.floor((21 - g) / 11))
  const j = (y + Math.floor(y / 4) + i + 2 - c + Math.floor(c / 4)) % 7
  const p = i - j
  const d = 1 + (p + 27 + Math.floor((p + 6) / 40)) % 31
  const m = 3 + Math.floor((p + 26) / 30)
  return new Date(y, m - 1, d)
}

function offsetDate(date, { years = 0, months = 0, days = 0, hours = 0, minutes = 0, seconds = 0, milliseconds = 0 }) {
  return new Date(
    date.getFullYear() + years,
    date.getMonth() + months,
    date.getDate() + days,
    date.getHours() + hours,
    date.getMinutes() + minutes,
    date.getSeconds() + seconds,
    date.getMilliseconds() + milliseconds
  );
}

function calcEasterDates(year) {
  easterSunday = calcEasterSunday(year)
  return {
    maundyThursday: offsetDate(easterSunday, { days: -3 }),
    goodFriday: offsetDate(easterSunday, { days: -2 }),
    easterSunday,
    easterMonday: offsetDate(easterSunday, { days: 1 }),
    ascensionDay: offsetDate(easterSunday, { days: 39 }),
    whitsun: offsetDate(easterSunday, { days: 49 }),
    whitMonday: offsetDate(easterSunday, { days: 50 })
  }
}

function getNorwegianHolidays(year) {
  easterDates = calcEasterDates(year)
  fixedHolidays = {
    newYear: new Date(year, 0, 1),
    arbeidernesDag: new Date(year, 4, 1),
    grunnlovsDagen: new Date(year, 4, 17),
    julaften: new Date(year, 11, 24), // Not necessarily for all workplaces
    forsteJuledag: new Date(year, 11, 25),
    andreJuledag: new Date(year, 11, 26),
  }
  return { ...easterDates, ...fixedHolidays }
}

function isInWeekend(date) {
  return !(date.getDay() % 6)
}

function isBetweenInclusive(date, from, to) {
  return (from.getTime() <= date.getTime()) && (date.getTime() <= to.getTime())
}

function countBusinessDays(from, to, holidays = []) {
  fromTime = from.getTime()
  toTime = to.getTime()
  if (fromTime > toTime) {
    throw new Error('"from" date cannot be later than "to" date')
  }
  const days = 1 + Math.round((toTime - fromTime) / 86400000)
  const saturdays = Math.floor((from.getDay() + days) / 7)
  const fromIsSunday = from.getDay() === 0
  const toIsSaturday = to.getDay() === 6
  const saturdaysAndSundays = 2 * saturdays + fromIsSunday - toIsSaturday
  holidays = holidays.reduce((acc, date) => 
    !isBetweenInclusive(date, from, to) || isInWeekend(date) ? acc : acc + 1
  , 0)
  return days - saturdaysAndSundays - holidays
}

function getExpectedWorkedHours(from, to) {
  console.log('from :>> ', from);
  console.log('to :>> ', to);
}

from = new Date(2022, 3, 1)
to = new Date(2025, 3, 30)
getExpectedWorkedHours(from, to)