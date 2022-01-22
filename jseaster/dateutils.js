const DEFAULT_WORKDAYS = [
  'monday',
  'tuesday',
  'wednesday',
  'thursday',
  'friday'
]

const DAYS_TO_NUM = {
  'monday': 1,
  'tuesday': 2,
  'wednesday': 3,
  'thursday': 4,
  'friday': 5,
  'saturday': 6,
  'sunday': 0
}

const NUM_TO_DAYS = {
  1: 'monday',
  2: 'tuesday',
  3: 'wednesday',
  4: 'thursday',
  5: 'friday',
  6: 'saturday',
  0: 'sunday'
}

/**
 * Based on Python's dateutil easter implementation
 * @param {number} year 
 * @returns {Date} date of easter sunday at given year
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

/**
 * Offset given date
 * @param {Date} date 
 * @param {object} offsets
 * @returns {Date} new date with offsets
 */
function offsetDate(
  date,
  { years = 0, months = 0, days = 0, hours = 0, minutes = 0, seconds = 0, milliseconds = 0 }
) {
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
 * @returns
 */
function calcEasterDates(year) {
  const easterSunday = calcEasterSunday(year)
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
 * @returns
 */
function getNorwegianHolidays(year) {
  const easterDates = calcEasterDates(year)
  const fixedHolidays = {
    newYear: new Date(year, 0, 1),
    workersDay: new Date(year, 4, 1),
    independenceDay: new Date(year, 4, 17),
    christmasEve: new Date(year, 11, 24), // Not necessarily for all workplaces
    christmasDay: new Date(year, 11, 25), // Forste juledag
    boxingDay: new Date(year, 11, 26), // Andre jule dag
    // newYearsEve: new Date(year, 11, 31)
  }
  return { ...easterDates, ...fixedHolidays }
}

/**
 * @param {Date} date 
 * @returns
 */
function inWeekend(date) {
  return !(date.getDay() % 6)
}

function validateFromToDates(from, to, { fromName = 'from', toName = 'to' } = {}) {
  if (from.getTime() > to.getTime())
    throw new Error(`"${fromName}" date cannot be later than "${toName}" date`)
}

/**
 * Inclusive 'from' and 'to' dates
 * @param {Date} date 
 * @param {Date} from
 * @param {Date} to
 * @returns
 */
function isBetween(date, from, to) {
  validateFromToDates(from, to)
  return (from.getTime() <= date.getTime()) && (date.getTime() <= to.getTime())
}

/**
 * Calculate number of days and number of saturdays and sundays between 'from' and 'to' dates.
 * Returned days are total days, so you must subtract saturdaysAndSundays to get work days
 * @param {Date} from
 * @param {Date} to
 * @returns
 */
function countDays(from, to) {
  validateFromToDates(from, to)
  const days = 1 + Math.round((to.getTime() - from.getTime()) / 86400000)
  const fromDay = from.getDay()
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
 * @param {Iterable<Date>} holidays 
 * @returns 
 */
function countWorkDays(from, to, holidays = []) {
  validateFromToDates(from, to)
  const { days, saturdays, sundays } = countDays(from, to)

  // Use for-of to support generators and such
  let holidaysInBusinessDays = 0
  for (holiday of holidays)
    if (isBetween(holiday, from, to) && !inWeekend(holiday))
      holidaysInBusinessDays++

  return days - saturdays - sundays - holidaysInBusinessDays
}

/**
 * Aggregates array of objects.
 * @param {Array<object>} objects, array of objects with identical properties 
 * @param {Function} aggregator, function to aggregate values
 * @returns {object}
 */
function aggregate(objects, aggregator = (x, y) => x + y) {
  const keys = Reflect.ownKeys(objects[0])
  return objects.slice(1).reduce((acc, object) =>
    keys.forEach(key => { acc[key] = aggregator(acc[key], object[key]) }) || acc
    , { ...objects[0] })
}

/**
 * Generator returning norwegian holidays between 'from' and 'to' dates
 * @param {Date} from 
 * @param {Date} to 
 * @returns {Generator<Date, void, void>}
 */
function* norwegianHolidaysGenerator(from, to) {
  validateFromToDates(from, to)
  const fromYear = from.getFullYear()
  for (i of Array(to.getFullYear() - fromYear + 1).keys())
    for (date of Object.values(getNorwegianHolidays(i + fromYear)))
      if (isBetween(date, from, to))
        yield date
      else
        continue
}

/**
 * E.g. given monday to friday, return ['saturday', 'sunday']
 * @param {Array<Date>} days 
 * @returns
 */
function getComplementWeekdays(days) {
  const days = new Set(days)
  return Reflect.ownKeys(DAYS_TO_NUM).filter(day => !days.has(day))
}

/**
 * @param {Iterable<Date>} holidays 
 * @param {Array<Date>} workdays 
 * @returns
 */
function countHolidaysInWorkdays(holidays, workdays) {
  const workdaySet = new Set(workdays.map(day => DAYS_TO_NUM[day]))
  let holidaysInWorkdays = 0
  if (holidays)
    for (holiday of holidays)
      if (isBetween(holiday, referenceDate, to) && workdaySet.has(holiday.getDay()))
        holidaysInWorkdays++
  return holidaysInWorkdays
}

/**
 * Get today's date object with zeroed out hours, minutes, seconds and milliseconds
 * @returns
 */
function getToday() {
  const date = new Date()
  date.setHours(0, 0, 0, 0)
  return date
}

/**
 * @param {number} actualHours 
 * @param {Date} referenceDate 
 * @param {number} referenceBalance 
 * @param {Date} to 
 * @param {{workdays: array<string>, holidays?: iterable<Date>}} optionals
 * @returns flex balance
 */
function calcFlexBalance(
  actualHours,
  referenceDate,
  referenceBalance,
  {
    to = getToday(),
    workdays = DEFAULT_WORKDAYS,
    holidays = norwegianHolidaysGenerator(referenceDate, to),
    workHoursPerDay = 7.5
  } = {}
) {
  validateFromToDates(referenceDate, to, { fromName: 'referenceDate' })
  const { days: dayCount, ...weekdaysCounts } = countDays(referenceDate, to)
  const holidaysInWorkdays = countHolidaysInWorkdays(holidays, workdays)
  const offdaysCount = getComplementWeekdays(workdays).reduce((acc, day) => acc + weekdaysCounts[day], 0)
  const expectedHours = (dayCount - holidaysInWorkdays - offdaysCount) * workHoursPerDay
  return actualHours - expectedHours + referenceBalance
}

module.exports = {
  aggregate,
  calcEasterDates,
  calcEasterSunday,
  calcFlexBalance,
  countDays,
  countWorkDays,
  DAYS_TO_NUM,
  DEFAULT_WORKDAYS,
  getComplementWeekdays,
  getNorwegianHolidays,
  getToday,
  inWeekend,
  isBetween,
  norwegianHolidaysGenerator,
  NUM_TO_DAYS,
  offsetDate,
}

