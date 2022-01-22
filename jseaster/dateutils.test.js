const { expect } = require("@jest/globals")
const dateutils = require('./dateutils')

const daysIn = {
  january2022: {
    days: 31,
    monday: 5,
    tuesday: 4,
    wednesday: 4,
    thursday: 4,
    friday: 4,
    saturday: 5,
    sunday: 5
  },
  february2022: {
    days: 28,
    monday: 4,
    tuesday: 4,
    wednesday: 4,
    thursday: 4,
    friday: 4,
    saturday: 4,
    sunday: 4
  },
  march2022: {
    days: 31,
    monday: 4,
    tuesday: 5,
    wednesday: 5,
    thursday: 5,
    friday: 4,
    saturday: 4,
    sunday: 4
  },
  april2022: {
    days: 30,
    monday: 4,
    tuesday: 4,
    wednesday: 4,
    thursday: 4,
    friday: 5,
    saturday: 5,
    sunday: 4
  }
}

test('countDays for January 2022 is correct', () => {
  const from = new Date(2022, 0, 1)
  const to = new Date(2022, 0, 31)
  expect(dateutils.countDays(from, to)).toEqual(daysIn.january2022)
})

test('countDays for February 2022 is correct', () => {
  const from = new Date(2022, 1, 1)
  const to = new Date(2022, 1, 28)
  expect(dateutils.countDays(from, to)).toEqual(daysIn.february2022)
})

test('countDays for March 2022 is correct', () => {
  const from = new Date(2022, 2, 1)
  const to = new Date(2022, 2, 31)
  expect(dateutils.countDays(from, to)).toEqual(daysIn.march2022)
})

test('countDays for April 2022 is correct', () => {
  const from = new Date(2022, 3, 1)
  const to = new Date(2022, 3, 30)
  expect(dateutils.countDays(from, to)).toEqual(daysIn.april2022)
})

test('countDays for Jan 1 2022 to April 31 2022 is correct', () => {
  const from = new Date(2022, 0, 1)
  const to = new Date(2022, 3, 30)
  expect(dateutils.countDays(from, to)).toEqual(dateutils.aggregate(Object.values(daysIn)))
})
test('slowCountDays for January 2022 is correct', () => {
  const from = new Date(2022, 0, 1)
  const to = new Date(2022, 0, 31)
  expect(dateutils.slowCountDays(from, to)).toEqual(daysIn.january2022)
})

test('slowCountDays for February 2022 is correct', () => {
  const from = new Date(2022, 1, 1)
  const to = new Date(2022, 1, 28)
  expect(dateutils.slowCountDays(from, to)).toEqual(daysIn.february2022)
})

test('slowCountDays for March 2022 is correct', () => {
  const from = new Date(2022, 2, 1)
  const to = new Date(2022, 2, 31)
  expect(dateutils.slowCountDays(from, to)).toEqual(daysIn.march2022)
})

test('slowCountDays for April 2022 is correct', () => {
  const from = new Date(2022, 3, 1)
  const to = new Date(2022, 3, 30)
  expect(dateutils.slowCountDays(from, to)).toEqual(daysIn.april2022)
})

test('slowCountDays for Jan 1 2022 to April 31 2022 is correct', () => {
  const from = new Date(2022, 0, 1)
  const to = new Date(2022, 3, 30)
  expect(dateutils.slowCountDays(from, to)).toEqual(dateutils.aggregate(Object.values(daysIn)))
})

test('slowCountDays and countDays agrees', () => {
  randInt = max => Math.floor(Math.random() * max);
  for (let i = 0; i < 100; i++) {
    const from = new Date(1970 + randInt(100), randInt(11), randInt(31))
    const to = dateutils.offsetDate(from, { days: randInt(10) })
    expect(dateutils.slowCountDays(from, to)).toEqual(dateutils.countDays(from, to))
  }
})

test('calcEasterDates is correct for year 2022', () => {
  expect(dateutils.calcEasterDates(2022)).toEqual({
    palmSunday: new Date(2022, 3, 10),
    maundyThursday: new Date(2022, 3, 14),
    goodFriday: new Date(2022, 3, 15),
    easterSunday: new Date(2022, 3, 17),
    easterMonday: new Date(2022, 3, 18),
    ascensionDay: new Date(2022, 4, 26),
    whitsun: new Date(2022, 5, 5),
    whitMonday: new Date(2022, 5, 6)
  })
})

test('getComplementWeekdays works', () => {
  expect(dateutils.getComplementWeekdays(['saturday', 'sunday'])).toEqual([
    'monday',
    'tuesday',
    'wednesday',
    'thursday',
    'friday'
  ])
})

test('calcFlexBalance case 1 ', () => {
  const referenceDate = new Date(2022, 0, 1)
  const referenceBalance = 12.5
  const to = new Date(2022, 0, 31)
  const expectedWorkDays = 21
  const expectedHoursPerDay = 7.5
  const actualHours = 160
  const balance = dateutils.calcFlexBalance(actualHours, referenceDate, referenceBalance, { to })
  expect(balance).toEqual(actualHours - expectedWorkDays * expectedHoursPerDay + referenceBalance)
  expect(balance).toEqual(15)
})