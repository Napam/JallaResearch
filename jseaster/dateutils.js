
/**
 * Based on Python's dateutil easter implementation
 * @param {number} year 
 * @returns {object} date of easter sunday at given year
 */
function calcEasterSunday(year) {
  if (typeof year !== 'number') {
    throw new Error('Year must specified as a number')
  }

  y = year
  g = y % 19
  c = Math.floor(y / 100)
  h = (c - Math.floor(c / 4) - Math.floor((8 * c + 13) / 25) + 19 * g + 15) % 30
  i = h - Math.floor(h / 28) * (1 - Math.floor(h / 28) * Math.floor(29 / (h + 1)) * Math.floor((21 - g) / 11))
  j = (y + Math.floor(y / 4) + i + 2 - c + Math.floor(c / 4)) % 7
  p = i - j
  d = 1 + (p + 27 + Math.floor((p + 6) / 40)) % 31
  m = 3 + Math.floor((p + 26) / 30)
  return new Date(y, m - 1, d)
}

/**
 * Offset given date
 * @param {Date} date 
 * @param {object} offsets
 * @returns {Date} new date with offsets
 */
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

/**
 * Returns easter days respective to given year
 * @param {number} year >= 1970
 * @returns {object}
 */
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

function inWeekend(date) {
  return !(date.getDay() % 6)
}

/**
 * Inclusive from to dates
 * @param {Date} date 
 * @param {Date} from 
 * @param {Date} to 
 * @returns {boolean}
 */
function isBetween(date, from, to) {
  return (from.getTime() <= date.getTime()) && (date.getTime() <= to.getTime())
}

/**
 * Calculate number of days and number of saturdays and sundays between 'from' and 'to' dates.
 * Returned days are total days, so you must subtract saturdaysAndSundays to get work days
 * @param {Date} from
 * @param {Date} to
 * @returns {{days: number, saturdaysAndSundays: number}}
 */
function countDays(from, to) {
  fromTime = from.getTime()
  toTime = to.getTime()
  if (fromTime > toTime)
    throw new Error('"from" date cannot be later than "to" date')

  days = 1 + Math.round((toTime - fromTime) / 86400000)
  fromDay = from.getDay()
  return {
    days,
    mondays: Math.floor((days + (fromDay + 5) % 7) / 7),
    tuesdays: Math.floor((days + (fromDay + 4) % 7) / 7),
    wednesdays: Math.floor((days + (fromDay + 3) % 7) / 7),
    thursdays: Math.floor((days + (fromDay + 2) % 7) / 7),
    fridays: Math.floor((days + (fromDay + 1) % 7) / 7),
    saturdays: Math.floor((days + (fromDay + 0) % 7) / 7),
    sundays: Math.floor((days + (fromDay - 1) % 7) / 7)
  }
}

/**
 * Count business days. Holidays are to be specified.
 * @param {object} from 
 * @param {Date} to 
 * @param {iterable} holidays 
 * @returns 
 */
function countWorkDays(from, to, holidays = []) {
  ({ days, saturdaysAndSundays } = countDays(from, to))

  // Use for-of to support generators and such
  holidaysInBusinessDays = 0
  for (holiday of holidays)
    if (isBetween(holiday, from, to) && !inWeekend(holiday)) { holidaysInBusinessDays++ }

  return days - saturdaysAndSundays - holidaysInBusinessDays
}

function* norwegianHolidaysGenerator(from, to) {
  fromYear = from.getFullYear()
  toYear = to.getFullYear()
  for (i of Array(toYear - fromYear + 1).keys())
    for (date of Object.values(getNorwegianHolidays(i + fromYear)))
      yield date
}

/**
 * Aggregates array of objects
 * @param {array<object>} objects 
 * @param {function} valueAggregator
 * @returns {object}
 */
function aggregate(objects, valueAggregator = (x, y) => x + y) {
  keys = Reflect.ownKeys(objects[0])
  return objects.slice(1).reduce((acc, object) =>
    keys.forEach(key => { acc[key] += object[key] }) || acc
    , { ...objects[0] })
}

function calcFlextime(from, to, offset) {
  return countDays(from, to)
}

module.exports = {
  calcEasterSunday,
  offsetDate,
  calcEasterDates,
  getNorwegianHolidays,
  inWeekend,
  isBetween,
  countDays,
  countWorkDays,
  norwegianHolidaysGenerator,
  aggregate,
  calcFlextime
}

