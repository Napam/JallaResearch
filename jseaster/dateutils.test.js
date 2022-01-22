const { expect } = require("@jest/globals")
const dateutils = require('./dateutils')

daysIn = {
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
  from = new Date(2022, 0, 1)
  to = new Date(2022, 0, 31)
  expect(dateutils.countDays(from, to)).toEqual(daysIn.january2022)
})

test('countDays for February 2022 is correct', () => {
  from = new Date(2022, 1, 1)
  to = new Date(2022, 1, 28)
  expect(dateutils.countDays(from, to)).toEqual(daysIn.february2022)
})

test('countDays for March 2022 is correct', () => {
  from = new Date(2022, 2, 1)
  to = new Date(2022, 2, 31)
  expect(dateutils.countDays(from, to)).toEqual(daysIn.march2022)
})

test('countDays for April 2022 is correct', () => {
  from = new Date(2022, 3, 1)
  to = new Date(2022, 3, 30)
  expect(dateutils.countDays(from, to)).toEqual(daysIn.april2022)
})

test('countDays for Jan 1 2022 to April 31 2022 is correct', () => {
  from = new Date(2022, 0, 1)
  to = new Date(2022, 3, 30)
  expect(dateutils.countDays(from, to)).toEqual(dateutils.aggregate(Object.values(daysIn)))
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
  referenceDate = new Date(2022, 0, 1)
  referenceBalance = 12.5
  to = new Date(2022, 0, 31)
  expectedWorkDays = 21
  expectedHoursPerDay = 7.5
  actualHours = 160
  balance = dateutils.calcFlexBalance(actualHours, referenceDate, referenceBalance, { to })
  expect(balance).toEqual(actualHours - expectedWorkDays * expectedHoursPerDay + referenceBalance)
  expect(balance).toEqual(15)
})