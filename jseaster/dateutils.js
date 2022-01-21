const DEFAULT_WORKDAYS = [
  'mondays',
  'tuesdays',
  'wednesdays',
  'thursdays',
  'fridays'
]

const DAYS_TO_NUM = {
  'mondays': 1,
  'tuesdays': 2,
  'wednesdays': 3,
  'thursdays': 4,
  'fridays': 5,
  'saturdays': 6,
  'sundays': 0
}

const NUM_TO_DAYS = {
  1: 'mondays',
  2: 'tuesdays',
  3: 'wednesdays',
  4: 'thursdays',
  5: 'fridays',
  6: 'saturdays',
  0: 'sundays'
}

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
    palmSunday: offsetDate(easterSunday, { days: -7 }),
    maundyThursday: offsetDate(easterSunday, { days: -3 }),
    goodFriday: offsetDate(easterSunday, { days: -2 }),
    easterSunday,
    easterMonday: offsetDate(easterSunday, { days: 1 }),
    ascensionDay: offsetDate(easterSunday, { days: 39 }),
    whitsun: offsetDate(easterSunday, { days: 49 }),
    whitMonday: offsetDate(easterSunday, { days: 50 })
  }
}

/**
 * @param {number} year 
 * @returns {{[key: string]: Date}} object where keys are holiday names, and values as dates
 */
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
 * Inclusive 'from' and 'to' dates
 * @param {Date} date 
 * @param {Date} a 
 * @param {Date} b 
 * @returns {boolean}
 */
function isBetween(date, a, b) {
  return (a.getTime() <= date.getTime()) && (date.getTime() <= b.getTime())
}

/**
 * Calculate number of days and number of saturdays and sundays between 'from' and 'to' dates.
 * Returned days are total days, so you must subtract saturdaysAndSundays to get work days
 * @param {Date} from
 * @param {Date} to
 * @returns {{
 *  days: number,
 *  mondays: number,
 *  tuesdays: number,
 *  wednesdays: number,
 *  thursdays: number,
 *  fridays: number,
 *  saturdays: number,
 *  sundays: number
 * }}
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
    [NUM_TO_DAYS[1]]: Math.floor((days + (fromDay + 5) % 7) / 7),
    [NUM_TO_DAYS[2]]: Math.floor((days + (fromDay + 4) % 7) / 7),
    [NUM_TO_DAYS[3]]: Math.floor((days + (fromDay + 3) % 7) / 7),
    [NUM_TO_DAYS[4]]: Math.floor((days + (fromDay + 2) % 7) / 7),
    [NUM_TO_DAYS[5]]: Math.floor((days + (fromDay + 1) % 7) / 7),
    [NUM_TO_DAYS[6]]: Math.floor((days + fromDay % 7) / 7),
    [NUM_TO_DAYS[0]]: Math.floor((days + (fromDay - 1) % 7) / 7)
  }
}

/**
 * Count business days. Holidays are to be specified.
 * @param {object} from 
 * @param {Date} to 
 * @param {iterable<Date>} holidays 
 * @returns 
 */
function countWorkDays(from, to, holidays = []) {
  ({ days, saturdays, sundays } = countDays(from, to))

  // Use for-of to support generators and such
  holidaysInBusinessDays = 0
  for (holiday of holidays)
    if (isBetween(holiday, from, to) && !inWeekend(holiday))
      holidaysInBusinessDays++

  return days - saturdays - sundays - holidaysInBusinessDays
}

/**
 * Aggregates array of objects.
 * @param {array<object>} objects, array of objects with identical properties 
 * @param {function} aggregator, function to aggregate values
 * @returns {object}
 */
function aggregate(objects, aggregator = (x, y) => x + y) {
  keys = Reflect.ownKeys(objects[0])
  return objects.slice(1).reduce((acc, object) =>
    keys.forEach(key => { acc[key] = aggregator(acc[key], object[key]) }) || acc
    , { ...objects[0] })
}

/**
 * @param {Date} from 
 * @param {Date} to 
 * @returns {Generator<Date, void, void>}
 */
function* norwegianHolidaysGenerator(from, to) {
  to.setHours(0, 0, 0, 0) // Hack so it can be used in signature of calcFlexBalance
  fromYear = from.getFullYear()
  toYear = to.getFullYear()
  for (i of Array(toYear - fromYear + 1).keys())
    for (date of Object.values(getNorwegianHolidays(i + fromYear)))
      yield date
}

/**
 * @param {number} actualHours 
 * @param {Date} referenceDate 
 * @param {number} referenceBalance 
 * @param {Date} to 
 * @param {{workdays: array<string>, holidays?: iterable<Date>}} optionals
 * @returns {number} flex balance
 */
function calcFlexBalance(
  actualHours,
  referenceDate,
  referenceBalance,
  to,
  {
    workdays = DEFAULT_WORKDAYS,
    holidays = norwegianHolidaysGenerator(referenceDate, new Date())
  } = {}
) {
  today = new Date()
  today.setHours(0, 0, 0, 0)
  const { days, ...weekdays } = countDays(referenceDate, today)
  workdaySet = new Set(workdays.map(day => DAYS_TO_NUM[day]))
  // offdays = new Set(Reflect.ownKeys(NUM_TO_DAYS).filter(day => !workdaySet.has(parseInt(day))))

  holidaysInBusinessDays = 0
  if (holidays)
    for (holiday of holidays)
      if (isBetween(holiday, referenceDate, today) && workdaySet.has(holiday.getDay()))
        holidaysInBusinessDays++

  workdays.reduce((acc, workday) => acc + weekdays[workday], 0)
  expectedDaysOfWork = days - holidaysInBusinessDays
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
  DEFAULT_WORKDAYS,
  calcFlexBalance
}

